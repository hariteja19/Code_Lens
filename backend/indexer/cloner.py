"""
Git Repository Cloner.

Handles cloning GitHub repositories for indexing using shallow clone (depth=1)
to minimize download size — we only need the latest code, not the full history.
"""

import git
import os
import shutil
from pathlib import Path
from typing import Optional
import stat


def rmtree_onerror(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    Retries after adding write permission if the error is an access error.
    """
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


DEFAULT_CLONE_BASE = os.path.join(os.path.expanduser("~"), ".codelens", "repos")


class RepoCloner:
    """Handles cloning and managing local copies of GitHub repositories."""

    def __init__(self, clone_base: str = DEFAULT_CLONE_BASE):
        self.clone_base = clone_base
        os.makedirs(clone_base, exist_ok=True)

    def extract_repo_id(self, github_url: str) -> str:
        """
        Extract a unique repo ID (owner/repository) from a GitHub URL.

        Examples:
            "https://github.com/facebook/react.git" → "facebook/react"
            "git@github.com:facebook/react.git"     → "facebook/react"
        """
        url = github_url.rstrip('/')
        if url.endswith('.git'):
            url = url[:-4]

        parts = url.replace(':', '/').split('/')
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return url

    def get_clone_path(self, repo_id: str) -> str:
        """Get the local filesystem path for a cloned repo."""
        return os.path.join(self.clone_base, *repo_id.split('/'))

    def clone(
        self,
        github_url: str,
        repo_id: Optional[str] = None,
        force_reclone: bool = False
    ) -> tuple[str, str]:
        """
        Clone a GitHub repository locally using shallow clone (depth=1).

        Args:
            github_url: Full GitHub URL (HTTPS or SSH)
            repo_id: Optional override for the repo identifier
            force_reclone: If True, delete existing clone and re-clone

        Returns:
            Tuple of (clone_path, repo_id)

        Raises:
            git.GitCommandError: If clone fails (bad URL, network issues, etc.)
        """
        if repo_id is None:
            repo_id = self.extract_repo_id(github_url)

        clone_path = self.get_clone_path(repo_id)

        if os.path.exists(clone_path):
            if force_reclone:
                shutil.rmtree(clone_path, onerror=rmtree_onerror)
            else:
                return clone_path, repo_id

        os.makedirs(os.path.dirname(clone_path), exist_ok=True)

        git.Repo.clone_from(
            github_url,
            clone_path,
            depth=1,
            single_branch=True,
        )

        return clone_path, repo_id

    def cleanup(self, repo_id: str):
        """Remove a cloned repository from disk."""
        clone_path = self.get_clone_path(repo_id)
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path, onerror=rmtree_onerror)

    def list_source_files(
        self,
        repo_path: str,
        extensions: Optional[set[str]] = None
    ) -> list[Path]:
        """
        Walk the repo directory and return all source files.
        Skips common non-source directories (node_modules, .git, etc.)

        Args:
            repo_path: Path to the cloned repo
            extensions: Set of file extensions to include

        Returns:
            List of Path objects for matching source files
        """
        if extensions is None:
            extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.cpp', '.go'}

        from backend.indexer.chunker import should_skip_path

        source_files = []
        repo = Path(repo_path)

        for file_path in repo.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix not in extensions:
                continue
            rel_path = str(file_path.relative_to(repo))
            if should_skip_path(rel_path):
                continue
            source_files.append(file_path)

        return source_files
