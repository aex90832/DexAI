'use strict';

const express = require('express');

const { Dex } = require('@pkmn/dex');
const { Generations } = require('@pkmn/data');
const { TeamValidator, Teams } = require('@pkmn/sim');
const { Dex: SimDex } = require('@pkmn/sim');
const calc = require('@smogon/calc');

const PORT = parseInt(process.env.PORT || '8991', 10);

const gens = new Generations(Dex);

const app = express();
app.use(express.json({ limit: '1mb' }));


// ---------------------------------------------------------------------------
// health
// ---------------------------------------------------------------------------

app.get('/health', (req, res) => {
  const pkg = require('./package.json');
  res.json({
    status: 'ok',
    service: 'pokedex-sim',
    node: process.version,
    packages: {
      '@pkmn/dex': depVersion('@pkmn/dex'),
      '@pkmn/sim': depVersion('@pkmn/sim'),
      '@smogon/calc': depVersion('@smogon/calc'),
    },
    app_version: pkg.version,
  });
});

function depVersion(name) {
  try {
    return require(`${name}/package.json`).version;
  } catch (e) {
    return 'unknown';
  }
}


// ---------------------------------------------------------------------------
// POST /validate_team
//
// Body: { team: "<showdown export text>", format: "gen9ou" }
// Returns: { valid, problems[], format, export, species[] }
//
// This is the ground truth for legality. Showdown's own validator, same code
// that runs on the ladder. If it says no, the team is illegal — no argument.
// ---------------------------------------------------------------------------

app.post('/validate_team', (req, res) => {
  const { team, format = 'gen9ou' } = req.body || {};

  if (typeof team !== 'string' || !team.trim()) {
    return res.status(400).json({
      error: 'team is required and must be a Showdown export string',
    });
  }

  let parsed;
  try {
    parsed = Teams.import(team);
  } catch (e) {
    return res.status(400).json({
      valid: false,
      problems: [`Could not parse team: ${e.message}`],
      format,
    });
  }

  if (!parsed || !parsed.length) {
    return res.status(400).json({
      valid: false,
      problems: ['Team parsed to zero Pokemon — check the export format.'],
      format,
    });
  }

  let validator;
  try {
    validator = new TeamValidator(format);
  } catch (e) {
    return res.status(400).json({
      valid: false,
      problems: [`Unknown format "${format}": ${e.message}`],
      format,
    });
  }

  let problems;
  try {
    problems = validator.validateTeam(parsed);
  } catch (e) {
    return res.status(500).json({
      valid: false,
      problems: [`Validator error: ${e.message}`],
      format,
    });
  }

  res.json({
    valid: !problems,
    problems: problems || [],
    format,
    species: parsed.map((s) => s.species || s.name),
    export: Teams.export(parsed),
  });
});


// ---------------------------------------------------------------------------
// POST /calc_damage
//
// Body: {
//   gen: 9,
//   attacker: { name, level, item, ability, nature, evs, ivs, boosts, status },
//   defender: { ...same... },
//   move: { name, crit, hits },
//   field: { gameType, weather, terrain, isReflect, isLightScreen, ... }
// }
//
// Everything except names is optional. Defaults are level 100, neutral nature,
// zero EVs — state your assumptions when you report the result.
// ---------------------------------------------------------------------------

app.post('/calc_damage', (req, res) => {
  const body = req.body || {};
  const genNum = body.gen || 9;

  if (!body.attacker || !body.attacker.name) {
    return res.status(400).json({ error: 'attacker.name is required' });
  }
  if (!body.defender || !body.defender.name) {
    return res.status(400).json({ error: 'defender.name is required' });
  }
  if (!body.move || !body.move.name) {
    return res.status(400).json({ error: 'move.name is required' });
  }

  let gen;
  try {
    gen = calc.Generations.get(genNum);
  } catch (e) {
    return res.status(400).json({ error: `Unknown generation ${genNum}` });
  }

  try {
    const attacker = buildPokemon(gen, body.attacker);
    const defender = buildPokemon(gen, body.defender);

    const move = new calc.Move(gen, body.move.name, {
      isCrit: body.move.crit || false,
      hits: body.move.hits,
      useZ: body.move.useZ || false,
      useMax: body.move.useMax || false,
    });

    const field = new calc.Field(body.field || {});

    const result = calc.calculate(gen, attacker, defender, move, field);

    // desc() and kochance() can both throw on odd inputs (0 damage, fixed
    // damage moves, status moves). Never let that take down the endpoint.
    const description = safely(() => result.desc());
    const koChance = safely(() => result.kochance().text);
    const range = safely(() => result.range());

    const damage = Array.isArray(result.damage) ? result.damage : [result.damage];
    const maxHP = defender.maxHP();

    res.json({
      description,
      damage,
      range,
      percent_range: range
        ? [
            round2((range[0] / maxHP) * 100),
            round2((range[1] / maxHP) * 100),
          ]
        : null,
      ko_chance: koChance,
      defender_max_hp: maxHP,
      // Whoever answers this call still has to know WHO ACTS FIRST to
      // interpret a "both sides OHKO" result — the raw percentages alone are
      // silent on that, which was flagged directly as a gap: two mutual OHKOs
      // read identically regardless of which one actually happens.
      speed_order: (() => {
        try {
          const aSpe = attacker.stats.spe, dSpe = defender.stats.spe;
          if (aSpe === dSpe) return 'speed tie — priority and mechanics decide, not shown here';
          return aSpe > dSpe
            ? `${body.attacker.name} moves first (${aSpe} vs ${dSpe} Speed)`
            : `${body.defender.name} moves first (${dSpe} vs ${aSpe} Speed)`;
        } catch (e) { return null; }
      })(),
      assumptions: {
        gen: genNum,
        attacker_level: attacker.level,
        defender_level: defender.level,
        attacker_evs: body.attacker.evs || 'none specified (0)',
        defender_evs: body.defender.evs || 'none specified (0)',
        // These three were previously silent defaults a caller could not see
        // without reading this file directly — each was specifically flagged
        // as a source of numbers that looked precise but rested on an unstated
        // choice: attacker.item defaulting to nothing while unrelated to what
        // a real set would run, allies_fainted defaulting to 0 and silently
        // erasing Supreme Overlord / Defiant / Beast Boost entirely, and
        // single_hit_no_sequencing meaning recoil, recovery, weather and
        // multi-turn effects are never modeled even though the response can
        // read as if they might be.
        attacker_item: body.attacker.item || 'none specified',
        defender_item: body.defender.item || 'none specified',
        attacker_allies_fainted: body.attacker.alliesFainted || 0,
        defender_allies_fainted: body.defender.alliesFainted || 0,
        single_hit_no_sequencing: true,
        note: 'This is one hit, one turn, in isolation. It does not know who '
            + 'moves first (see speed_order above), does not apply recoil, '
            + 'recovery, weather, or multi-turn effects, and any ability whose '
            + 'effect depends on game state (fainted allies, prior stat drops, '
            + 'etc.) uses only what was explicitly passed in this request — '
            + 'never inferred from the matchup.',
      },
    });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

function buildPokemon(gen, spec) {
  return new calc.Pokemon(gen, spec.name, {
    level: spec.level || 100,
    item: spec.item,
    ability: spec.ability,
    nature: spec.nature,
    evs: spec.evs || {},
    ivs: spec.ivs || {},
    boosts: spec.boosts || {},
    status: spec.status,
    teraType: spec.teraType,
    isDynamaxed: spec.isDynamaxed || false,
    // Without this, calc.Pokemon defaults it to 0 silently — meaning every
    // Supreme Overlord calc returned the SAME number regardless of ability,
    // because the field @smogon/calc actually reads for that boost was never
    // being set. Confirmed via direct inspection: `alliesFainted` is a real
    // constructor field on calc.Pokemon; nothing here was ever passing it.
    alliesFainted: spec.alliesFainted || 0,
  });
}

function safely(fn) {
  try {
    return fn();
  } catch (e) {
    return null;
  }
}

function round2(n) {
  return Math.round(n * 100) / 100;
}


// ---------------------------------------------------------------------------
// GET /dump/:gen
//
// Full structured dex dump for a generation. Consumed once by ingest.py to
// build the SQLite tables — not called at query time.
//
// This is why the sidecar has to exist before ingest can run: @pkmn is the
// authoritative data layer and it's JavaScript only.
// ---------------------------------------------------------------------------

app.get('/dump/:gen', async (req, res) => {
  const genNum = parseInt(req.params.gen, 10);

  if (!(genNum >= 1 && genNum <= 9)) {
    return res.status(400).json({ error: 'gen must be 1-9' });
  }

  let gen;
  try {
    gen = gens.get(genNum);
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }

  try {
    const species = [];
    for (const s of gen.species) {
      species.push({
        id: s.id,
        name: s.name,
        num: s.num,
        types: s.types,
        base_stats: s.baseStats,
        bst: Object.values(s.baseStats).reduce((a, b) => a + b, 0),
        abilities: s.abilities,
        base_species: s.baseSpecies,
        forme: s.forme || null,
        other_formes: s.otherFormes || [],
        battle_only: !!s.battleOnly,
        required_item: Array.isArray(s.requiredItem) ? (s.requiredItem[0] || null) : (s.requiredItem || null),
        prevo: s.prevo || null,
        evos: s.evos || [],
        evo_level: s.evoLevel || null,
        evo_item: s.evoItem || null,
        evo_condition: s.evoCondition || null,
        evo_type: s.evoType || null,
        egg_groups: s.eggGroups || [],
        weight_kg: s.weightkg,
        height_m: s.heightm,
        tier: s.tier || null,
        doubles_tier: s.doublesTier || null,
        nfe: !!(s.evos && s.evos.length),
      });
    }

    // Learnsets are async and one call per species, but @pkmn caches after the
    // first, so this is fast. Keep the FULL source strings, not just move names:
    // "4L32" means gen 4, Level-up, level 32. Discarding those loses the answer
    // to "what level does this learn at", which is the main thing someone
    // replaying an older game wants to know.
    //
    // gen.learnsets.get() returns each species' OWN incremental learnset, not
    // the full effective learnset a real game would show — egg and tutor moves
    // in particular are frequently recorded ONLY on the base form of an
    // evolution line, on the documented assumption that callers resolve
    // inheritance themselves by walking prevo. Confirmed directly against this
    // library: Pawniard has Sucker Punch as a tutor move; Bisharp and Kingambit
    // both report `false` for it despite legitimately being able to learn it in
    // every real game and on every published competitive set. Left unpatched,
    // this silently drops moves for any fully-evolved Pokemon whose earlier
    // forms carry egg/tutor/event moves not repeated on the final stage — which
    // is a common pattern, not a rare one.
    //
    // Fix: after collecting each species' own learnset, walk its prevo chain
    // and merge in every ancestor's moves too. A move already known at this
    // species keeps ITS OWN sources (do not overwrite); a move only known via
    // an ancestor gets that ancestor's sources, so "how do I get this move" for
    // Kingambit correctly points at whatever generation/method let Pawniard
    // learn it, rather than fabricating a Kingambit-specific source that never
    // existed.
    const learnsets = {};
    // gen.species is @pkmn/data's own iterable collection type, not a plain
    // array — it has no .map(). Materialize it once via the spread, matching
    // how the rest of this file already consumes it with for-of.
    const allSpecies = [...gen.species];
    const speciesById = new Map(allSpecies.map(s => [s.id, s]));

    for (const s of allSpecies) {
      const ls = await gen.learnsets.get(s.id);
      learnsets[s.id] = (ls && ls.learnset) ? { ...ls.learnset } : {};
    }

    for (const s of allSpecies) {
      let cur = s;
      const seen = new Set([cur.id]);   // guard against a malformed prevo cycle
      while (cur.prevo) {
        const prevoId = typeof cur.prevo === 'string'
          ? cur.prevo.toLowerCase().replace(/[^a-z0-9]/g, '')
          : null;
        if (!prevoId || seen.has(prevoId)) break;
        seen.add(prevoId);
        const ancestorLearnset = learnsets[prevoId];
        if (ancestorLearnset) {
          for (const [moveId, sources] of Object.entries(ancestorLearnset)) {
            if (!(moveId in learnsets[s.id])) {
              learnsets[s.id][moveId] = sources;
            }
          }
        }
        cur = speciesById.get(prevoId);
        if (!cur) break;
      }
    }

    const moves = [];
    for (const m of gen.moves) {
      moves.push({
        id: m.id,
        name: m.name,
        num: m.num,
        type: m.type,
        category: m.category,
        base_power: m.basePower,
        accuracy: m.accuracy,
        pp: m.pp,
        priority: m.priority,
        target: m.target,
        flags: m.flags,
        short_desc: m.shortDesc,
        desc: m.desc,
        is_z: m.isZ || null,
        is_max: m.isMax || null,
      });
    }

    const abilities = [];
    for (const a of gen.abilities) {
      abilities.push({
        id: a.id,
        name: a.name,
        num: a.num,
        short_desc: a.shortDesc,
        desc: a.desc,
      });
    }

    const items = [];
    for (const i of gen.items) {
      items.push({
        id: i.id,
        name: i.name,
        num: i.num,
        short_desc: i.shortDesc,
        desc: i.desc,
        mega_evolves: i.megaEvolves || null,
        is_berry: !!i.isBerry,
        is_choice: !!i.isChoice,
      });
    }

    // Raw Showdown type chart encoding:
    //   0 = normal (1x), 1 = weak (2x), 2 = resist (0.5x), 3 = immune (0x)
    // Keyed by ATTACKING type, read from the DEFENDING type's entry.
    const typechart = {};
    const simGen = SimDex.forGen(genNum);
    for (const t of simGen.types.all()) {
      typechart[t.name] = t.damageTaken || {};
    }

    res.json({
      gen: genNum,
      generated_at: new Date().toISOString(),
      counts: {
        species: species.length,
        moves: moves.length,
        abilities: abilities.length,
        items: items.length,
        learnsets: Object.keys(learnsets).length,
      },
      species,
      learnsets,
      moves,
      abilities,
      items,
      typechart,
    });
  } catch (e) {
    console.error('dump failed:', e);
    res.status(500).json({ error: e.message, stack: e.stack });
  }
});


// ---------------------------------------------------------------------------
// GET /formats
//
// List of format IDs the validator accepts. Useful for checking what to pass
// to /validate_team, and for the model to know what metagames exist.
// ---------------------------------------------------------------------------

app.get('/formats', (req, res) => {
  try {
    const formats = SimDex.formats
      .all()
      .filter((f) => f.effectType === 'Format')
      .map((f) => ({
        id: f.id,
        name: f.name,
        gen: f.gen,
        game_type: f.gameType,
        section: f.section || null,
      }));
    res.json({ count: formats.length, formats });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});


// ---------------------------------------------------------------------------
// POST /compare_teams
//
// Body: { team_a, team_b, format?, gen?, label_a?, label_b? }
//
// Deterministic structural comparison of two teams. Speed order, shared
// weaknesses, and a full threat matrix from real damage calculations.
//
// Explicitly NOT a prediction. There is no simulation here and no outcome data,
// so no win probability is produced — the response says so, every time, because
// a language model handed a matchup will otherwise invent a percentage.
// ---------------------------------------------------------------------------

const TYPE_CODE = { 0: 1, 1: 2, 2: 0.5, 3: 0 };   // Showdown damageTaken encoding

function typeMultiplier(simGen, attackingType, defendingTypes) {
  let mult = 1;
  for (const dt of defendingTypes) {
    const info = simGen.types.get(dt);
    if (!info || !info.damageTaken) continue;
    const code = info.damageTaken[attackingType];
    if (code !== undefined && TYPE_CODE[code] !== undefined) mult *= TYPE_CODE[code];
  }
  return mult;
}

function buildMember(gen, simGen, set) {
  const name = set.species || set.name;
  const mon = new calc.Pokemon(gen, name, {
    level: set.level || 100,
    item: set.item,
    ability: set.ability,
    nature: set.nature,
    evs: set.evs || {},
    ivs: set.ivs || {},
    teraType: set.teraType,
  });

  let spe;
  try {
    spe = mon.stats.spe;
  } catch (e) {
    spe = null;
  }

  const species = simGen.species.get(name);
  const types = (species && species.types) || mon.types || [];

  return {
    name,
    calcMon: mon,
    types: Array.from(types),
    ability: set.ability || null,
    item: set.item || null,
    tera: set.teraType || null,
    level: set.level || 100,
    moves: (set.moves || []).filter(Boolean),
    spe,
    // Choice Scarf is common enough that ignoring it makes the speed order wrong.
    speEffective: (set.item === 'Choice Scarf' && spe) ? Math.floor(spe * 1.5) : spe,
    scarfed: set.item === 'Choice Scarf',
  };
}

function threatMatrix(gen, attackers, defenders) {
  const out = [];
  for (const atk of attackers) {
    for (const def of defenders) {
      let best = null;
      for (const moveName of atk.moves) {
        let move;
        try {
          move = new calc.Move(gen, moveName);
        } catch (e) {
          continue;
        }
        if (!move.bp) continue;                   // status move
        try {
          const r = calc.calculate(gen, atk.calcMon, def.calcMon, move);
          const dmg = Array.isArray(r.damage) ? r.damage : [r.damage];
          const max = Math.max(...dmg.flat().filter(n => typeof n === 'number'));
          const maxHP = def.calcMon.maxHP();
          const pct = Math.round((max / maxHP) * 1000) / 10;
          if (!best || pct > best.max_pct) {
            let ko = null;
            try { ko = r.kochance().text; } catch (e) { /* calc can throw here */ }
            best = { move: moveName, max_pct: pct, ko };
          }
        } catch (e) {
          continue;
        }
      }
      if (best) {
        out.push({
          attacker: atk.name,
          defender: def.name,
          move: best.move,
          max_pct: best.max_pct,
          ko: best.ko,
          ohko: best.max_pct >= 100,
        });
      }
    }
  }
  return out;
}

function sharedWeaknesses(simGen, members) {
  const byType = {};
  for (const t of simGen.types.all()) {
    const weak = members.filter(m => typeMultiplier(simGen, t.name, m.types) >= 2);
    if (weak.length >= 2) {
      byType[t.name] = { count: weak.length, members: weak.map(m => m.name) };
    }
  }
  return Object.entries(byType)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 8)
    .map(([type, v]) => ({ type, count: v.count, members: v.members }));
}

app.post('/compare_teams', (req, res) => {
  const body = req.body || {};
  const genNum = body.gen || 9;
  const labelA = body.label_a || 'Team A';
  const labelB = body.label_b || 'Team B';

  if (!body.team_a || !body.team_b) {
    return res.status(400).json({ error: 'team_a and team_b are both required' });
  }

  let gen, simGen;
  try {
    gen = calc.Generations.get(genNum);
    simGen = SimDex.forGen(genNum);
  } catch (e) {
    return res.status(400).json({ error: `Unknown generation ${genNum}` });
  }

  let setsA, setsB;
  try {
    setsA = Teams.import(body.team_a);
    setsB = Teams.import(body.team_b);
  } catch (e) {
    return res.status(400).json({ error: `Could not parse a team: ${e.message}` });
  }
  if (!setsA || !setsA.length || !setsB || !setsB.length) {
    return res.status(400).json({ error: 'One of the teams parsed to zero Pokemon.' });
  }

  try {
    const A = setsA.map(s => buildMember(gen, simGen, s));
    const B = setsB.map(s => buildMember(gen, simGen, s));

    const speedOrder = [...A.map(m => ({ ...m, team: labelA })),
                        ...B.map(m => ({ ...m, team: labelB }))]
      .filter(m => m.speEffective != null)
      .sort((a, b) => b.speEffective - a.speEffective)
      .map(m => ({
        name: m.name, team: m.team, speed: m.speEffective,
        note: m.scarfed ? 'Choice Scarf (1.5x applied)' : undefined,
      }));

    const aToB = threatMatrix(gen, A, B);
    const bToA = threatMatrix(gen, B, A);

    const threatenedInB = new Set(aToB.filter(t => t.max_pct >= 100).map(t => t.defender));
    const threatenedInA = new Set(bToA.filter(t => t.max_pct >= 100).map(t => t.defender));

    const strip = m => ({
      name: m.name, types: m.types, ability: m.ability, item: m.item,
      tera: m.tera, speed: m.speEffective, moves: m.moves,
    });

    res.json({
      format: body.format || `gen${genNum}ou`,
      gen: genNum,
      teams: {
        [labelA]: { size: A.length, members: A.map(strip) },
        [labelB]: { size: B.length, members: B.map(strip) },
      },
      speed_order: speedOrder,
      shared_weaknesses: {
        [labelA]: sharedWeaknesses(simGen, A),
        [labelB]: sharedWeaknesses(simGen, B),
      },
      threats: {
        [`${labelA} -> ${labelB}`]: aToB.sort((x, y) => y.max_pct - x.max_pct).slice(0, 24),
        [`${labelB} -> ${labelA}`]: bToA.sort((x, y) => y.max_pct - x.max_pct).slice(0, 24),
      },
      summary: {
        [`${labelA} can OHKO`]: `${threatenedInB.size} of ${B.length}`,
        [`${labelB} can OHKO`]: `${threatenedInA.size} of ${A.length}`,
        fastest: speedOrder.length ? speedOrder[0] : null,
      },
      assumptions: {
        note: 'Damage uses the sets as written. Unspecified EVs default to 0 and '
            + 'unspecified natures to neutral, so damage may be understated for '
            + 'incomplete sets. No field conditions, weather, terrain, hazards, '
            + 'or boosts are applied. Tera is set on the Pokemon object but not '
            + 'guaranteed to affect every calculation path — verify indepen'
            + 'dently for a Tera-critical matchup rather than trusting this by '
            + 'default. Abilities whose effect depends on live battle state — '
            + 'Supreme Overlord (fainted allies), Beast Boost, Intimidate '
            + 'already applied, prior stat drops for Defiant/Competitive — are '
            + 'evaluated at their BASE state (0 fainted allies, 0 boosts) '
            + 'because a team export has no field for battle-in-progress state. '
            + 'A team built around a late-game Supreme Overlord sweep will '
            + 'show its damage numbers at their floor, not their realistic '
            + 'late-game value.',
      },
      disclaimer:
        'STRUCTURAL COMPARISON ONLY. This is not a prediction and contains no '
        + 'win probability. No battle was simulated and there is no outcome data '
        + 'behind these numbers. Do not state or estimate a percentage chance of '
        + 'either team winning — describe the matchup qualitatively using the '
        + 'speed order, shared weaknesses, and threat matrix above.',
    });
  } catch (e) {
    console.error('compare_teams failed:', e);
    res.status(500).json({ error: e.message, stack: e.stack });
  }
});


// ---------------------------------------------------------------------------
// POST /review_team
//
// Body: { team, format?, gen? }
//
// The beginner's actual question: "is my team any good?" Same machinery as
// compare_teams, pointed at one team, plus role coverage — the structural gaps
// that sink new players' teams before any matchup matters.
// ---------------------------------------------------------------------------

const ROLE_MOVES = {
  'hazard removal': ['Rapid Spin', 'Defog', 'Court Change', 'Mortal Spin', 'Tidy Up'],
  'entry hazards': ['Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web', 'Ceaseless Edge',
                    'Stone Axe'],
  'recovery': ['Recover', 'Roost', 'Soft-Boiled', 'Slack Off', 'Synthesis', 'Moonlight',
               'Morning Sun', 'Rest', 'Shore Up', 'Milk Drink', 'Wish', 'Strength Sap'],
  'speed control': ['Thunder Wave', 'Icy Wind', 'Sticky Web', 'Tailwind', 'Trick Room',
                    'Glare', 'Electroweb', 'Bulldoze'],
  'pivoting': ['U-turn', 'Volt Switch', 'Flip Turn', 'Teleport', 'Parting Shot',
               'Baton Pass', 'Shed Tail', 'Chilly Reception'],
  'status': ['Will-O-Wisp', 'Toxic', 'Thunder Wave', 'Spore', 'Sleep Powder',
             'Glare', 'Nuzzle'],
  'setup': ['Swords Dance', 'Nasty Plot', 'Dragon Dance', 'Calm Mind', 'Bulk Up',
            'Quiver Dance', 'Shell Smash', 'Agility', 'Rock Polish', 'Iron Defense',
            'Victory Dance', 'Tidy Up'],
};

app.post('/review_team', (req, res) => {
  const body = req.body || {};
  const genNum = body.gen || 9;

  if (!body.team) return res.status(400).json({ error: 'team is required' });

  let gen, simGen;
  try {
    gen = calc.Generations.get(genNum);
    simGen = SimDex.forGen(genNum);
  } catch (e) {
    return res.status(400).json({ error: `Unknown generation ${genNum}` });
  }

  let sets;
  try {
    sets = Teams.import(body.team);
  } catch (e) {
    return res.status(400).json({ error: `Could not parse team: ${e.message}` });
  }
  if (!sets || !sets.length) {
    return res.status(400).json({ error: 'Team parsed to zero Pokemon.' });
  }

  try {
    const members = sets.map(s => buildMember(gen, simGen, s));
    const allMoves = new Set(members.flatMap(m => m.moves));

    // Which structural roles are covered, and by whom.
    const roles = {};
    for (const [role, moves] of Object.entries(ROLE_MOVES)) {
      const covered = members.filter(m => m.moves.some(mv => moves.includes(mv)));
      roles[role] = {
        covered: covered.length > 0,
        by: covered.map(m => m.name),
      };
    }

    // Offensive coverage: which types does the team hit super-effectively?
    const attackingTypes = new Set();
    for (const m of members) {
      for (const moveName of m.moves) {
        try {
          const mv = new calc.Move(gen, moveName);
          if (mv.bp) attackingTypes.add(mv.type);
        } catch (e) { /* unknown move name */ }
      }
    }
    const unresisted = [];
    for (const t of simGen.types.all()) {
      const hit = [...attackingTypes].some(
        at => typeMultiplier(simGen, at, [t.name]) >= 2);
      if (!hit) unresisted.push(t.name);
    }

    const speeds = members
      .filter(m => m.speEffective != null)
      .sort((a, b) => b.speEffective - a.speEffective)
      .map(m => ({ name: m.name, speed: m.speEffective,
                   note: m.scarfed ? 'Choice Scarf (1.5x applied)' : undefined }));

    const weaknesses = sharedWeaknesses(simGen, members);

    const gaps = [];
    for (const [role, v] of Object.entries(roles)) {
      if (!v.covered && ['hazard removal', 'entry hazards', 'recovery',
                         'speed control'].includes(role)) {
        gaps.push(`No ${role}.`);
      }
    }
    for (const w of weaknesses) {
      if (w.count >= 3) {
        gaps.push(`${w.count} members weak to ${w.type}: ${w.members.join(', ')}.`);
      }
    }
    if (members.length < 6) gaps.push(`Only ${members.length} Pokemon on the team.`);

    res.json({
      format: body.format || `gen${genNum}ou`,
      gen: genNum,
      size: members.length,
      members: members.map(m => ({
        name: m.name, types: m.types, ability: m.ability,
        item: m.item, tera: m.tera, speed: m.speEffective, moves: m.moves,
      })),
      speed_tiers: speeds,
      shared_weaknesses: weaknesses,
      roles,
      offensive_gaps: {
        attacking_types: [...attackingTypes],
        types_not_hit_super_effectively: unresisted,
      },
      structural_gaps: gaps.length ? gaps : ['No obvious structural gaps found.'],
      note: 'Structural review only. It does not judge whether individual sets are '
          + 'good, and it cannot tell you how the team performs in practice. Run '
          + 'validate_team separately to confirm legality.',
    });
  } catch (e) {
    console.error('review_team failed:', e);
    res.status(500).json({ error: e.message, stack: e.stack });
  }
});


app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: err.message });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`pokedex-sim listening on 0.0.0.0:${PORT}`);
});
