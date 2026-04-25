"""
goal_detector.py
────────────────
Détecte les buts via la projection 2D (Tactical Map) et attribue :
  - le but au dernier possesseur (scorer)
  - la passe décisive à l'avant-dernier possesseur de la MÊME équipe (assister)

Système de coordonnées (ViewTransformer) :
  X : 0 → 23.32 m  (longueur filmée, gauche → droite)
  Y : 0 → 68 m     (largeur du terrain, haut → bas)
"""

# ── Dimensions FIFA ────────────────────────────────────────────────────────────
FIELD_WIDTH = 68.0   # m
GOAL_WIDTH  = 7.32   # m
GOAL_DEPTH  = 2.0    # m — tolérance de détection vers l'intérieur du terrain

_goal_y_left  = (FIELD_WIDTH - GOAL_WIDTH) / 2   # 30.34 m
_goal_y_right = _goal_y_left + GOAL_WIDTH         # 37.66 m

_X_MIN = 0.0
_X_MAX = 23.32   # ViewTransformer.court_length

# ── Zones de but (polygones en mètres) ────────────────────────────────────────
GOAL_ZONES: dict[str, list[tuple[float, float]]] = {

    # Côté gauche — équipe 2 marque dans ce but
    'team_2_scores': [
        (_X_MIN - GOAL_DEPTH, _goal_y_left),
        (_X_MIN + GOAL_DEPTH, _goal_y_left),
        (_X_MIN + GOAL_DEPTH, _goal_y_right),
        (_X_MIN - GOAL_DEPTH, _goal_y_right),
    ],

    # Côté droit — équipe 1 marque dans ce but
    'team_1_scores': [
        (_X_MAX - GOAL_DEPTH, _goal_y_left),
        (_X_MAX + GOAL_DEPTH, _goal_y_left),
        (_X_MAX + GOAL_DEPTH, _goal_y_right),
        (_X_MAX - GOAL_DEPTH, _goal_y_right),
    ],
}


# ── Intersection géométrique : Ray Casting ────────────────────────────────────

def point_in_polygon(
    point:   tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    """
    Teste si (x, y) est à l'intérieur d'un polygone quelconque via l'algorithme
    du Ray Casting (Jordan curve theorem). Complexité : O(n).

    Pour chaque arête (vi → vj) :
      1. Le rayon horizontal depuis (px, py) franchit-il l'arête en Y ?
      2. Si oui, l'abscisse d'intersection est-elle à droite de px ?
      Chaque croisement inverse l'état "inside".
    """
    px, py = point
    n      = len(polygon)
    inside = False
    j      = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        crosses_y   = (yi > py) != (yj > py)
        x_intersect = xj + (py - yj) / (yi - yj) * (xi - xj)
        if crosses_y and px < x_intersect:
            inside = not inside

        j = i

    return inside


def point_in_any_goal(
    point:      tuple[float, float],
    goal_zones: dict = GOAL_ZONES,
) -> str | None:
    """Retourne la clé de la zone si le point y est inclus, sinon None."""
    for zone_name, polygon in goal_zones.items():
        if point_in_polygon(point, polygon):
            return zone_name
    return None


# ── GoalDetector ──────────────────────────────────────────────────────────────

class GoalDetector:
    """
    Détecte les buts à partir des positions 2D projetées du ballon.
    Gère le cooldown anti-doublon, l'attribution du but et de la passe décisive.

    Logique d'assist :
      On recherche dans possession_history l'entrée PRÉCÉDANT le scorer
      dont l'équipe est la même. Ce joueur reçoit l'assist.
    """

    ZONE_TO_SCORING_TEAM: dict[str, int] = {
        'team_1_scores': 1,
        'team_2_scores': 2,
    }

    def __init__(
        self,
        fps:              int   = 24,
        cooldown_seconds: float = 5.0,
        goal_zones:       dict  = GOAL_ZONES,
    ):
        self.fps             = fps
        self.cooldown_frames = int(fps * cooldown_seconds)
        self.goal_zones      = goal_zones
        self._last_goal_frame: int = -(self.cooldown_frames + 1)

    # ── Recherche de l'assisteur ───────────────────────────────────────────────

    @staticmethod
    def _find_assister(
        possession_history: list[dict],
        scorer_id:          int,
        scorer_team:        int,
    ) -> int:
        """
        Remonte l'historique de possession pour trouver l'avant-dernier joueur
        de la MÊME équipe que le buteur (et différent du buteur lui-même).

        Algorithme :
          - Parcourt possession_history à rebours depuis la fin
          - Saute le scorer lui-même (reprise de possession)
          - Retourne le premier joueur trouvé avec la même équipe
          - Retourne -1 si aucun candidat

        Args:
            possession_history : liste ordonnée [{player_id, team, frame}, ...]
            scorer_id          : player_id du buteur
            scorer_team        : team_id du buteur

        Returns:
            player_id de l'assisteur, ou -1
        """
        # On cherche en partant de la fin, en ignorant les occurrences du scorer
        seen_scorer = False
        for entry in reversed(possession_history):
            pid  = entry['player_id']
            team = entry['team']

            if pid == scorer_id:
                seen_scorer = True   # on a trouvé le scorer, on continue
                continue

            if seen_scorer and team == scorer_team:
                return pid           # premier joueur différent, même équipe

        return -1

    # ── Point d'entrée principal ───────────────────────────────────────────────

    def check(
        self,
        ball_pos_2d:        tuple[float, float] | None,
        frame_num:          int,
        scorer_id:          int,
        scorer_team:        int,
        player_stats:       dict,
        events:             list,
        possession_history: list[dict] | None = None,
    ) -> str | None:
        """
        Vérifie si le ballon est dans une cage et déclenche GOAL + ASSIST.

        Args:
            ball_pos_2d        : (x, y) projeté en mètres, None si hors champ
            frame_num          : numéro de la frame courant
            scorer_id          : dernier possesseur confirmé
            scorer_team        : équipe du scorer
            player_stats       : dict partagé {player_id: {'buts': 0, ...}}
            events             : liste d'événements partagée
            possession_history : liste [{player_id, team, frame}] pour les assists

        Returns:
            Clé de la zone si but validé, None sinon.
        """
        if ball_pos_2d is None:
            return None

        zone = point_in_any_goal(ball_pos_2d, self.goal_zones)
        if zone is None:
            return None

        # Cooldown
        if (frame_num - self._last_goal_frame) < self.cooldown_frames:
            return None

        self._last_goal_frame = frame_num
        scoring_team = self.ZONE_TO_SCORING_TEAM.get(zone, -1)

        # ── Initialisation des stats si nécessaire ─────────────────────────
        def _ensure(pid: int) -> None:
            if pid not in player_stats:
                player_stats[pid] = {
                    'passes_reussies': 0, 'passes_tentees': 0,
                    'interceptions':   0, 'buts': 0, 'assists': 0,
                }

        # ── Attribution du but ─────────────────────────────────────────────
        if scorer_id != -1:
            _ensure(scorer_id)
            player_stats[scorer_id]['buts'] += 1

        # ── Attribution de la passe décisive ──────────────────────────────
        assister_id = -1
        if possession_history and scorer_id != -1:
            assister_id = self._find_assister(
                possession_history, scorer_id, scorer_team
            )
            if assister_id != -1:
                _ensure(assister_id)
                player_stats[assister_id]['assists'] += 1
                print(
                    f"[ASSIST]          Frame {frame_num:>4d} | "
                    f"Joueur {assister_id} (équipe {scorer_team}) → "
                    f"passe décisive pour le but de joueur {scorer_id}"
                )

        # ── Enregistrement de l'événement ─────────────────────────────────
        event = {
            'type':         'GOAL',
            'frame':        frame_num,
            'zone':         zone,
            'scoring_team': scoring_team,
            'scorer_id':    scorer_id,
            'assister_id':  assister_id,
            'ball_pos_2d':  ball_pos_2d,
        }
        events.append(event)

        print(
            f"[BUT !!!]         Frame {frame_num:>4d} | "
            f"Équipe {scoring_team} marque ! "
            f"Buteur : joueur {scorer_id}"
            + (f", Assist : joueur {assister_id}" if assister_id != -1 else "")
            + f" — pos 2D ({ball_pos_2d[0]:.2f}m, {ball_pos_2d[1]:.2f}m)"
        )
        return zone
