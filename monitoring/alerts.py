import json
import os
import time
from datetime import datetime

ALERTS_FILE = "alerts.json"
MAX_ALERTS = 100


SEVERITY_LEVELS = {
    "info":    1,
    "warning": 2,
    "error":   3,
    "critical": 4,
}


def _load_alerts():
    if not os.path.exists(ALERTS_FILE):
        return []
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_alerts(alerts):
    try:
        with open(ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(alerts[-MAX_ALERTS:], f, indent=2)
    except OSError as e:
        print(f"[ALERTS] Failed to save: {e}")


class AlertManager:
    def __init__(self):
        self._alerts = _load_alerts()

    def trigger_alert(self, alert_type, message, severity="warning",
                      camera_id=None, details=None):
        alert = {
            "id":        f"{alert_type}_{int(time.time())}",
            "type":      alert_type,
            "message":   message,
            "severity":  severity,
            "camera_id": camera_id,
            "details":   details or {},
            "timestamp":  time.time(),
            "datetime":  datetime.now().isoformat(),
            "resolved":  False,
        }
        self._alerts.append(alert)
        self._alerts = self._alerts[-MAX_ALERTS:]
        _save_alerts(self._alerts)

        sev = severity.upper()
        print(f"[ALERT] [{sev}] {alert_type}: {message}")

        # Optional webhook
        webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
        if webhook_url:
            self._send_webhook(webhook_url, alert)

        return alert["id"]

    def resolve_alert(self, alert_type, camera_id=None):
        """Mark alerts of a type as resolved."""
        updated = 0
        for alert in self._alerts:
            if (alert["type"] == alert_type
                    and alert.get("camera_id") == camera_id
                    and not alert["resolved"]):
                alert["resolved"] = True
                alert["resolved_at"] = time.time()
                updated += 1
        if updated:
            _save_alerts(self._alerts)
            print(f"[ALERT] Resolved {updated} alert(s) of type '{alert_type}'")

    def get_active_alerts(self, severity_filter=None):
        """Get unresolved alerts, optionally filtered by severity."""
        active = [a for a in self._alerts if not a.get("resolved", False)]
        if severity_filter:
            active = [a for a in active if a["severity"] == severity_filter]
        return sorted(active, key=lambda x: x["timestamp"], reverse=True)

    def get_all_alerts(self, limit=50):
        """Get all alerts (including resolved), most recent first."""
        return sorted(self._alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def _send_webhook(self, url, alert):
        """Send alert to webhook URL (Slack/Discord compatible)."""
        try:
            import urllib.request
            payload = json.dumps({
                "text": f"[{alert['severity'].upper()}] {alert['type']}: {alert['message']}",
                "alert": alert,
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[ALERT] Webhook sent successfully")
        except Exception as e:
            print(f"[ALERT] Webhook failed: {e}")


def trigger_alert(alert_type, message, severity="warning",
                  camera_id=None, details=None):
    """Convenience function to trigger an alert."""
    from monitoring import get_alert_manager
    return get_alert_manager().trigger_alert(
        alert_type, message, severity, camera_id, details
    )


# Singleton instance
_alert_manager = None


def get_alert_manager():
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
