import sys
from enum import Enum, auto

sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


# ── Configuration ──────────────────────────────────────────────────────────────
BALL_PROXIMITY_THRESHOLD = 60   # pixels – distance balle→pieds pour être "proche"
MIN_POSSESSION_FRAMES    = 3    # frames consécutives pour valider la possession
FREE_BALL_FRAMES         = 5    # frames consécutives sans possesseur pour passer en FREE_BALL
# ───────────────────────────────────────────────────────────────────────────────


class BallState(Enum):
    """États possibles du ballon dans la machine à états."""
    FREE_BALL          = auto()   # ballon non contrôlé
    ATTACHED_TO_PLAYER = auto()   # ballon sous contrôle d'un joueur


def _default_stats() -> dict:
    return {
        'passes_reussies': 0,
        'passes_tentees':  0,
        'interceptions':   0,
        'buts':            0,
        'assists':         0,
    }


class PlayerBallAssigner:
    """
    Gère la possession individuelle du ballon et détecte les événements
    (passes réussies, interceptions) via une machine à états.

    États :
        FREE_BALL          – ballon libre (aucun joueur sous le seuil depuis FREE_BALL_FRAMES)
        ATTACHED_TO_PLAYER – ballon contrôlé par current_ball_possessor

    Transitions :
        ATTACHED(A) → FREE_BALL        : A devient last_possessor
        FREE_BALL   → ATTACHED(B)      : compare équipe(A) vs équipe(B)
            - même équipe, A ≠ B  → passes_reussies[A]++, passes_tentees[A]++
            - équipes adverses    → interceptions[B]++, passes_tentees[A]++ (passe ratée)
            - même joueur A == B  → reprise de balle (pas d'événement)
    """

    def __init__(
        self,
        proximity_threshold: int = BALL_PROXIMITY_THRESHOLD,
        min_possession_frames: int = MIN_POSSESSION_FRAMES,
        free_ball_frames: int = FREE_BALL_FRAMES,
    ):
        self.proximity_threshold   = proximity_threshold
        self.min_possession_frames = min_possession_frames
        self.free_ball_frames      = free_ball_frames

        # ── État courant de la machine à états ──────────────────────────────
        self.ball_state: BallState = BallState.FREE_BALL
        self.current_ball_possessor: int = -1   # joueur officiellement en possession
        self.last_possessor: int         = -1   # dernier joueur avant FREE_BALL
        self.last_possessor_team: int    = -1   # équipe du last_possessor

        # ── Debouncing – acquisition de possession ──────────────────────────
        self._candidate_id:     int = -1
        self._candidate_frames: int = 0

        # ── Debouncing – libération du ballon ───────────────────────────────
        self._free_frames_count: int = 0   # frames consécutives sans personne proche

        # ── Statistiques individuelles ──────────────────────────────────────
        # Initialisé à la volée dès qu'un player_id est rencontré
        self.player_stats: dict[int, dict] = {}

        # ── Journal d'événements ────────────────────────────────────────────
        self.events: list[dict] = []

        # ── Historique de possession ordonné ────────────────────────────────
        # Chaque entrée : {'player_id': int, 'team': int, 'frame': int}
        # Utilisé par GoalDetector pour attribuer les passes décisives
        self.possession_history: list[dict] = []

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ensure_player(self, player_id: int) -> None:
        """Crée l'entrée de stats pour un joueur si elle n'existe pas encore."""
        if player_id not in self.player_stats:
            self.player_stats[player_id] = _default_stats()

    @staticmethod
    def _get_ball_center(ball_bbox) -> tuple:
        return get_center_of_bbox(ball_bbox)

    @staticmethod
    def _get_foot_position(player_bbox) -> tuple:
        x1, y1, x2, y2 = player_bbox
        return (int((x1 + x2) / 2), int(y2))

    # ── Détection du joueur le plus proche ────────────────────────────────────

    def assign_ball_to_player(self, players: dict, ball_bbox) -> int:
        """
        Retourne le player_id du joueur dont les pieds sont les plus proches
        du centre du ballon, sous BALL_PROXIMITY_THRESHOLD. Sinon -1.
        """
        ball_center    = self._get_ball_center(ball_bbox)
        best_player_id = -1
        min_dist       = float('inf')

        for player_id, player in players.items():
            foot = self._get_foot_position(player['bbox'])
            dist = measure_distance(foot, ball_center)
            if dist < self.proximity_threshold and dist < min_dist:
                min_dist       = dist
                best_player_id = player_id

        return best_player_id

    # ── Machine à états + détection d'événements ──────────────────────────────

    def _on_attach(self, new_possessor: int, new_team: int, frame_num: int) -> None:
        """
        Appelé quand le ballon passe de FREE_BALL → ATTACHED_TO_PLAYER(new_possessor).
        Analyse la transition pour détecter passe réussie ou interception.
        """
        self._ensure_player(new_possessor)

        if self.last_possessor != -1 and self.last_possessor != new_possessor:
            self._ensure_player(self.last_possessor)
            same_team = (self.last_possessor_team == new_team) and (self.last_possessor_team != -1)

            if same_team:
                # ── Passe réussie ──────────────────────────────────────────
                self.player_stats[self.last_possessor]['passes_reussies'] += 1
                self.player_stats[self.last_possessor]['passes_tentees']  += 1
                event = {
                    'type':   'PASSE_REUSSIE',
                    'frame':  frame_num,
                    'from':   self.last_possessor,
                    'to':     new_possessor,
                    'team':   new_team,
                }
                self.events.append(event)
                print(
                    f"[PASSE RÉUSSIE]   Frame {frame_num:>4d} | "
                    f"Joueur {self.last_possessor} → Joueur {new_possessor} "
                    f"(équipe {new_team})"
                )
            else:
                # ── Interception / passe ratée ─────────────────────────────
                self.player_stats[self.last_possessor]['passes_tentees']  += 1
                self.player_stats[new_possessor]['interceptions']         += 1
                event = {
                    'type':          'INTERCEPTION',
                    'frame':         frame_num,
                    'lost_by':       self.last_possessor,
                    'lost_by_team':  self.last_possessor_team,
                    'gained_by':     new_possessor,
                    'gained_by_team': new_team,
                }
                self.events.append(event)
                print(
                    f"[INTERCEPTION]    Frame {frame_num:>4d} | "
                    f"Joueur {new_possessor} (équipe {new_team}) intercepte "
                    f"de Joueur {self.last_possessor} (équipe {self.last_possessor_team})"
                )

        # Transition d'état
        self.ball_state             = BallState.ATTACHED_TO_PLAYER
        self.current_ball_possessor = new_possessor
        self._free_frames_count     = 0

        # Enregistrement dans l'historique de possession
        self.possession_history.append({
            'player_id': new_possessor,
            'team':      new_team,
            'frame':     frame_num,
        })

    def _on_release(self, frame_num: int, possessor_team: int) -> None:
        """
        Appelé quand le ballon passe de ATTACHED_TO_PLAYER → FREE_BALL.
        Mémorise le dernier possesseur.
        """
        self.last_possessor      = self.current_ball_possessor
        self.last_possessor_team = possessor_team
        self.ball_state          = BallState.FREE_BALL
        self.current_ball_possessor = -1
        print(
            f"[BALLON LIBRE]    Frame {frame_num:>4d} | "
            f"Joueur {self.last_possessor} (équipe {self.last_possessor_team}) "
            f"perd le ballon"
        )

    # ── Point d'entrée principal ───────────────────────────────────────────────

    def update_possession(
        self,
        players: dict,
        ball_bbox,
        frame_num: int,
        player_teams: dict | None = None,   # {player_id: team_id} pour la frame courante
    ) -> int:
        """
        Met à jour la machine à états frame par frame.

        Args:
            players:      dict {player_id: {'bbox': [...], ...}} de la frame
            ball_bbox:    bbox du ballon [x1, y1, x2, y2]
            frame_num:    numéro de la frame courante
            player_teams: dict {player_id: team_id} — si None, l'équipe est lue
                          depuis players[id].get('team', -1)

        Retourne:
            current_ball_possessor (int, -1 si FREE_BALL)
        """
        candidate = self.assign_ball_to_player(players, ball_bbox)

        # ── Résolution de l'équipe du candidat ────────────────────────────
        def _team_of(pid: int) -> int:
            if player_teams and pid in player_teams:
                return player_teams[pid]
            if pid in players:
                return players[pid].get('team', -1)
            return -1

        # ── Debouncing – acquisition ───────────────────────────────────────
        if candidate != -1:
            self._free_frames_count = 0   # quelqu'un est proche → reset compteur libre

            if candidate == self._candidate_id:
                self._candidate_frames += 1
            else:
                self._candidate_id     = candidate
                self._candidate_frames = 1

            candidate_confirmed = (self._candidate_frames >= self.min_possession_frames)

            if candidate_confirmed:
                if self.ball_state == BallState.FREE_BALL:
                    # FREE_BALL → ATTACHED
                    self._on_attach(candidate, _team_of(candidate), frame_num)

                elif self.ball_state == BallState.ATTACHED_TO_PLAYER:
                    if candidate != self.current_ball_possessor:
                        # Changement de possesseur direct (contact immédiat)
                        # On passe d'abord en FREE_BALL pour déclencher la logique passe/interception
                        self._on_release(frame_num, _team_of(self.current_ball_possessor))
                        self._on_attach(candidate, _team_of(candidate), frame_num)
                    # Sinon : même joueur → rien à faire

        else:
            # Personne proche sur cette frame
            self._candidate_id     = -1
            self._candidate_frames = 0

            if self.ball_state == BallState.ATTACHED_TO_PLAYER:
                self._free_frames_count += 1
                if self._free_frames_count >= self.free_ball_frames:
                    # Transition ATTACHED → FREE_BALL après N frames sans contact
                    self._on_release(frame_num, _team_of(self.current_ball_possessor))

        return self.current_ball_possessor

    # ── Résumé des statistiques ────────────────────────────────────────────────

    def print_stats_summary(self) -> None:
        """Affiche un tableau de synthèse des statistiques de chaque joueur."""
        print("\n" + "=" * 65)
        print(f"{'RÉSUMÉ DES STATISTIQUES INDIVIDUELLES':^65}")
        print("=" * 65)
        header = f"{'Joueur':>8} | {'Passes ✓':>8} | {'Tentées':>7} | {'Interc.':>7} | {'Buts':>6} | {'Assists':>7}"
        print(header)
        print("-" * 65)
        for pid in sorted(self.player_stats):
            s = self.player_stats[pid]
            print(
                f"{pid:>8} | {s['passes_reussies']:>8} | {s['passes_tentees']:>7} | "
                f"{s['interceptions']:>7} | {s['buts']:>6} | {s['assists']:>7}"
            )
        print("=" * 65)
        print(f"  Total événements détectés : {len(self.events)}")
        passes   = sum(1 for e in self.events if e['type'] == 'PASSE_REUSSIE')
        intercep = sum(1 for e in self.events if e['type'] == 'INTERCEPTION')
        print(f"  Passes réussies : {passes}  |  Interceptions : {intercep}")
        print("=" * 65 + "\n")