"""
stats_exporter.py
─────────────────
Agrège les statistiques individuelles des joueurs depuis :
  - player_stats       : {player_id: {'buts', 'passes_reussies', ...}}  (PlayerBallAssigner)
  - tracks['players']  : contient 'team', 'speed' et 'distance' par frame (SpeedAndDistance_Estimator)

Et exporte un fichier JSON structuré par player_id.

Format de sortie :
{
  "generated_at": "2026-04-25T16:20:00",
  "video_source":  "08fd33_4.mp4",
  "total_frames":  751,
  "players": {
    "7": {
      "player_id":       7,
      "team":            1,
      "buts":            1,
      "assists":         0,
      "passes_reussies": 4,
      "passes_tentees":  5,
      "interceptions":   0,
      "vitesse_max_kmh": 28.34,
      "distance_totale_m": 312.7
    },
    ...
  }
}
"""

import json
from datetime import datetime


# ── Agrégation vitesse/distance depuis les tracks ─────────────────────────────

def _aggregate_speed_distance(tracks: dict) -> dict[int, dict]:
    """
    Parcourt tous les frames de tracks['players'] et, pour chaque player_id,
    collecte la vitesse max et la distance totale parcourue à la dernière frame
    dans laquelle il apparaît (SpeedAndDistance_Estimator cumule 'distance').

    Returns:
        {player_id: {'vitesse_max_kmh': float, 'distance_totale_m': float}}
    """
    player_frames: dict[int, dict] = {}   # {pid: {'speed_max': float, 'dist': float}}

    for frame_data in tracks.get('players', []):
        for pid, info in frame_data.items():
            speed    = info.get('speed',    None)
            distance = info.get('distance', None)

            if pid not in player_frames:
                player_frames[pid] = {'speed_max': 0.0, 'dist_last': 0.0}

            if speed    is not None:
                player_frames[pid]['speed_max']  = max(player_frames[pid]['speed_max'], speed)
            if distance is not None:
                player_frames[pid]['dist_last']  = distance   # cumulatif → on prend la dernière valeur

    return {
        pid: {
            'vitesse_max_kmh':   round(v['speed_max'], 2),
            'distance_totale_m': round(v['dist_last'], 2),
        }
        for pid, v in player_frames.items()
    }


def _get_player_team(tracks: dict, player_id: int) -> int:
    """Retourne l'équipe du joueur en cherchant dans les premiers frames."""
    for frame_data in tracks.get('players', []):
        if player_id in frame_data:
            return frame_data[player_id].get('team', -1)
    return -1


# ── Export principal ──────────────────────────────────────────────────────────

def export_stats_to_json(
    player_stats: dict,
    tracks:       dict,
    output_path:  str  = 'output_videos/player_stats.json',
    video_source: str  = 'unknown',
) -> dict:
    """
    Fusionne player_stats (PlayerBallAssigner) avec vitesse/distance (tracks)
    et exporte un fichier JSON structuré.

    Args:
        player_stats : {player_id: {'buts', 'assists', 'passes_reussies', ...}}
        tracks       : dict complet du pipeline (contient 'players' avec speed/distance)
        output_path  : chemin du fichier JSON à écrire
        video_source : nom du fichier vidéo source (metadata)

    Returns:
        Le dict exporté (utile pour tests ou affichage direct).
    """
    speed_dist = _aggregate_speed_distance(tracks)
    total_frames = len(tracks.get('players', []))

    # Collecte tous les player_ids connus (union des deux sources)
    all_pids = set(player_stats.keys()) | set(speed_dist.keys())

    players_output: dict[str, dict] = {}

    for pid in sorted(all_pids):
        stats = player_stats.get(pid, {})
        perf  = speed_dist.get(pid, {})
        team  = _get_player_team(tracks, pid)

        players_output[str(pid)] = {
            'player_id':          pid,
            'team':               team,
            'buts':               stats.get('buts',            0),
            'assists':            stats.get('assists',          0),
            'passes_reussies':    stats.get('passes_reussies', 0),
            'passes_tentees':     stats.get('passes_tentees',  0),
            'interceptions':      stats.get('interceptions',   0),
            'vitesse_max_kmh':    perf.get('vitesse_max_kmh',  0.0),
            'distance_totale_m':  perf.get('distance_totale_m', 0.0),
        }

    output = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'video_source': video_source,
        'total_frames': total_frames,
        'players':      players_output,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[EXPORT] Statistiques écrites dans : {output_path}")
    print(f"         {len(players_output)} joueurs | {total_frames} frames analysées")

    return output
