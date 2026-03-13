-- =============================================================
-- NBA 2K ML Project — Database Schema
-- =============================================================

CREATE TABLE IF NOT EXISTS players (
    player_id       INTEGER PRIMARY KEY,
    full_name       TEXT    NOT NULL,
    first_name      TEXT,
    last_name       TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seasons (
    season_year     INTEGER PRIMARY KEY,
    season_string   TEXT NOT NULL,
    game_version    TEXT NOT NULL
);

INSERT INTO seasons (season_year, season_string, game_version) VALUES
    (2019, '2018-19', 'nba-2k20'),
    (2020, '2019-20', 'nba-2k21'),
    (2021, '2020-21', 'nba-2k22'),
    (2022, '2021-22', 'nba-2k23'),
    (2023, '2022-23', 'nba-2k24'),
    (2024, '2023-24', 'nba-2k25'),
    (2025, '2024-25', 'nba-2k26')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS ratings (
    id              SERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(player_id),
    season_year     INTEGER REFERENCES seasons(season_year),
    ovr_rating      INTEGER NOT NULL,
    team_in_game    TEXT,
    rank_in_game    TEXT,
    is_rookie       BOOLEAN DEFAULT FALSE,
    scraped_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, season_year)
);

CREATE TABLE IF NOT EXISTS stats (
    id              SERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(player_id),
    season_year     INTEGER REFERENCES seasons(season_year),
    team_abbr       TEXT,
    gp              INTEGER,
    min             NUMERIC(5,2),
    pts             NUMERIC(5,2),
    reb             NUMERIC(5,2),
    ast             NUMERIC(5,2),
    stl             NUMERIC(5,2),
    blk             NUMERIC(5,2),
    tov             NUMERIC(5,2),
    fg_pct          NUMERIC(5,3),
    fg3_pct         NUMERIC(5,3),
    ft_pct          NUMERIC(5,3),
    net_rating      NUMERIC(6,2),
    usg_pct         NUMERIC(5,3),
    ast_pct         NUMERIC(5,3),
    ast_to          NUMERIC(5,2),
    pie             NUMERIC(5,3),
    fetched_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, season_year)
);

CREATE TABLE IF NOT EXISTS features (
    id              SERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(player_id),
    season_year     INTEGER REFERENCES seasons(season_year),
    age             INTEGER,
    career_year     INTEGER,
    pts_delta       NUMERIC(5,2),
    reb_delta       NUMERIC(5,2),
    ast_delta       NUMERIC(5,2),
    ovr_prev        INTEGER,
    ovr_delta       INTEGER,
    is_rookie       BOOLEAN DEFAULT FALSE,
    position_enc    TEXT,
    UNIQUE(player_id, season_year)
);

CREATE TABLE IF NOT EXISTS name_corrections (
    id              SERIAL PRIMARY KEY,
    hoopshype_name  TEXT NOT NULL,
    nba_api_name    TEXT NOT NULL,
    player_id       INTEGER REFERENCES players(player_id),
    confidence      NUMERIC(4,2),
    manually_verified BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ratings_player   ON ratings(player_id);
CREATE INDEX IF NOT EXISTS idx_ratings_season   ON ratings(season_year);
CREATE INDEX IF NOT EXISTS idx_stats_player     ON stats(player_id);
CREATE INDEX IF NOT EXISTS idx_stats_season     ON stats(season_year);
CREATE INDEX IF NOT EXISTS idx_features_player  ON features(player_id);
CREATE INDEX IF NOT EXISTS idx_features_season  ON features(season_year);

CREATE OR REPLACE VIEW ml_dataset AS
SELECT
    r.player_id,
    p.full_name,
    r.season_year,
    s.game_version,
    r.ovr_rating,
    r.is_rookie,
    st.gp, st.min, st.pts, st.reb, st.ast, st.stl, st.blk, st.tov,
    st.fg_pct, st.fg3_pct, st.ft_pct,
    st.net_rating, st.usg_pct, st.ast_pct, st.ast_to, st.pie,
    f.age, f.career_year,
    f.pts_delta, f.reb_delta, f.ast_delta,
    f.ovr_prev, f.ovr_delta,
    CASE WHEN r.season_year <= 2023 THEN 'train' ELSE 'test' END AS split
FROM ratings r
JOIN players  p  ON p.player_id   = r.player_id
JOIN seasons  s  ON s.season_year = r.season_year
LEFT JOIN stats st ON st.player_id = r.player_id AND st.season_year = r.season_year
LEFT JOIN features f ON f.player_id = r.player_id AND f.season_year = r.season_year;