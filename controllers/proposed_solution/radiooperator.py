import json


def build_victim_message(robot, current_x, current_y, confidence):
    """
    Builds the exact JSON dict required by the rules.
    Uses calculated odometry instead of hardcoding [0, 0, 0].
    """
    return {
        "timestamp": robot.getTime(),
        "robot_id": robot.getName(),
        "position": [current_x, current_y, 0.0],  # Z is roughly 0
        "victim_found": True,
        "victim_confidence": confidence,
    }


def build_ready_message(robot_id):
    """
    Small handshake message robot1 sends once its offline artifacts
    (e.g. the targets CSV) are fully written to disk. robot2 waits for
    this instead of assuming a fixed number of timesteps is enough.
    """
    return {"type": "ready", "robot_id": robot_id}


def rapid_fire_burst(emitter, message, burst_count=5):
    """
    Sends the same message multiple times rapidly to help ensure delivery.
    NOTE: check your competition/sim rules before keeping burst_count > 1 --
    some scoring harnesses penalize duplicate victim reports rather than
    rewarding redundant sends.
    """
    if emitter is None:
        return

    payload = json.dumps(message).encode()
    for _ in range(burst_count):
        emitter.send(payload)
