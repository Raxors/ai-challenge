"""
HTTP-клиент для JetBrains YouTrack REST API.

Использует только стандартную библиотеку (urllib.request).
"""

import json
import urllib.request
import urllib.parse
import urllib.error


class YouTrackAPI:
    """Клиент для YouTrack REST API."""

    def __init__(self, base_url, token):
        """
        Args:
            base_url: URL инстанса YouTrack (например https://example.youtrack.cloud)
            token:    Permanent token (perm:...)
        """
        self.base_url = base_url.rstrip("/")
        self.token = token

    # ── Публичные методы ──────────────────────────────────

    def get_projects(self, top=10):
        """Получить список проектов."""
        params = {"fields": "id,name,shortName,description", "$top": str(top)}
        return self._request("GET", "/api/admin/projects", params=params)

    def get_issues(self, query=None, project=None, top=10, skip=0):
        """Поиск задач."""
        fields = (
            "id,idReadable,summary,description,created,resolved,"
            "reporter(login),customFields(name,value(name))"
        )
        params = {"fields": fields, "$top": str(top), "$skip": str(skip)}
        if query:
            params["query"] = query
        if project:
            if "query" in params:
                params["query"] = f"project: {project} " + params["query"]
            else:
                params["query"] = f"project: {project}"
        return self._request("GET", "/api/issues", params=params)

    def get_issue(self, issue_id):
        """Получить одну задачу по ID (например 'PROJ-123')."""
        fields = (
            "id,idReadable,summary,description,created,"
            "reporter(login),comments(text,author(login)),"
            "customFields(name,value(name))"
        )
        params = {"fields": fields}
        return self._request("GET", f"/api/issues/{issue_id}", params=params)

    def create_issue(self, project_id, summary, description=None):
        """Создать задачу в проекте."""
        body = {"project": {"id": project_id}, "summary": summary}
        if description:
            body["description"] = description
        params = {"fields": "id,idReadable,summary"}
        return self._request("POST", "/api/issues", params=params, body=body)

    def update_issue(self, issue_id, summary=None, description=None):
        """Обновить задачу."""
        body = {}
        if summary is not None:
            body["summary"] = summary
        if description is not None:
            body["description"] = description
        params = {"fields": "id,idReadable,summary"}
        return self._request("POST", f"/api/issues/{issue_id}", params=params, body=body)

    def delete_issue(self, issue_id):
        """Удалить задачу."""
        return self._request("DELETE", f"/api/issues/{issue_id}")

    # ── Внутренний HTTP ───────────────────────────────────

    def _request(self, method, path, params=None, body=None):
        """Выполнить HTTP-запрос к YouTrack API."""
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = resp.read().decode("utf-8")
                if resp_data:
                    return json.loads(resp_data)
                return None
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"YouTrack API error {e.code}: {e.reason}. {error_body}"
            )
        except urllib.error.URLError as e:
            raise RuntimeError(f"YouTrack connection error: {e.reason}")
