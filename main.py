from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner, GoalDetector, export_stats_to_json
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # ── 1. Lecture de la vidéo ────────────────────────────────────────────────
    video_frames = read_video('input_videos/match_quartier.mp4')

    # ── 2. Tracking ───────────────────────────────────────────────────────────
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path='stubs/track_stubs.pkl'
    )
    tracker.add_position_to_tracks(tracks)

    # ── 3. Estimation du mouvement de caméra ─────────────────────────────────
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # ── 4. Transformation de vue ──────────────────────────────────────────────
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ── 5. Interpolation des positions du ballon ──────────────────────────────
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ── 6. Vitesse & distance ─────────────────────────────────────────────────
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ── 7. Attribution des équipes ────────────────────────────────────────────
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team']       = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # ── 8. Possession individuelle + machine à états (passes / interceptions / buts) ──
    player_assigner = PlayerBallAssigner()
    goal_detector   = GoalDetector(fps=24, cooldown_seconds=5.0)
    team_ball_control = []

    print("\n=== Suivi de possession & détection d'événements ===")

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']

        # Construit le dict {player_id: team} pour cette frame
        player_teams = {
            pid: info.get('team', -1)
            for pid, info in player_track.items()
        }

        # Mise à jour de la machine à états (debouncing + événements)
        current_possessor = player_assigner.update_possession(
            player_track,
            ball_bbox,
            frame_num,
            player_teams=player_teams,
        )

        # Marque visuellement le possesseur confirmé
        if current_possessor != -1 and current_possessor in tracks['players'][frame_num]:
            tracks['players'][frame_num][current_possessor]['has_ball'] = True

        # ── Détection de but via position projetée 2D du ballon ──────────
        ball_track = tracks['ball'][frame_num].get(1, {})
        ball_pos_2d = ball_track.get('position_transformed', None)
        if ball_pos_2d is not None:
            ball_pos_2d = tuple(ball_pos_2d)   # (x_metres, y_metres)

        # Buteur = possesseur actuel, ou dernier possesseur si FREE_BALL
        scorer_id   = (current_possessor if current_possessor != -1
                       else player_assigner.last_possessor)
        scorer_team = (player_teams.get(scorer_id, -1)
                       if scorer_id != -1 else -1)

        goal_detector.check(
            ball_pos_2d        = ball_pos_2d,
            frame_num          = frame_num,
            scorer_id          = scorer_id,
            scorer_team        = scorer_team,
            player_stats       = player_assigner.player_stats,
            events             = player_assigner.events,
            possession_history = player_assigner.possession_history,
        )

        # Pour la barre de possession d'équipe (rétrocompatibilité)
        if current_possessor != -1 and current_possessor in tracks['players'][frame_num]:
            team = tracks['players'][frame_num][current_possessor].get('team', -1)
            if team != -1:
                team_ball_control.append(team)
            elif team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
        elif team_ball_control:
            team_ball_control.append(team_ball_control[-1])
        else:
            team_ball_control.append(0)

    team_ball_control = np.array(team_ball_control)

    # Nombre total de buts
    total_goals = sum(1 for e in player_assigner.events if e['type'] == 'GOAL')
    print(f"\n=== Buts détectés : {total_goals} ===")

    # Affiche le résumé des statistiques individuelles
    player_assigner.print_stats_summary()

    # ── 11. Export JSON ────────────────────────────────────────────
    export_stats_to_json(
        player_stats = player_assigner.player_stats,
        tracks       = tracks,
        output_path  = 'output_videos/player_stats.json',
        video_source = 'input_videos/match_quartier.mp4',
    )

    # ── 9. Rendu vidéo ────────────────────────────────────────────────────────
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # ── 10. Sauvegarde ────────────────────────────────────────────────────────
    save_video(output_video_frames, 'output_videos/output_video11.avi')


if __name__ == '__main__':
    main()