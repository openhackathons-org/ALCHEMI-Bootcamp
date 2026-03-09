"""Notebook-friendly API client for ALCHEMI BMD/BGR endpoints."""

import requests

from .models import MDAtomicData, MDConfig, MDRequest, MDReply, AtomicData, BGRRequest, BGRReply
from .cache import cache_exists, load_cache, save_cache


def check_endpoint(server_url: str, timeout: int = 5) -> bool:
    """Return True if the ALCHEMI endpoint health-check responds 200."""
    url = f"{server_url}/v2/health/ready"
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.InvalidURL):
        return False


def run_md(
    mdatoms: MDAtomicData,
    mdconfig: MDConfig,
    server_url: str,
    timeout: int = 300,
) -> MDReply:
    """Submit an MD request and return the parsed reply.

    Unlike the CLI script this does *not* loop-wait; it raises on failure.
    """
    url = f"{server_url}/infer"
    payload = MDRequest(atoms=mdatoms, config=mdconfig).model_dump()
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    reply = MDReply(**response.json())
    if reply.status != "Success":
        raise RuntimeError(f"MD simulation failed: {reply.status} — {reply.info}")
    return reply


def run_bgr(
    atoms_list: list[AtomicData],
    server_url: str,
    cellopt: bool = False,
    opttol: float | None = None,
    timeout: int = 300,
) -> BGRReply:
    """Submit a BGR request and return the parsed reply."""
    url = f"{server_url}/infer"
    payload = BGRRequest(atoms=atoms_list, cellopt=cellopt, opttol=opttol).model_dump()
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    reply = BGRReply(**response.json())
    if reply.status != "Success":
        raise RuntimeError(f"BGR failed: {reply.status} — {reply.info}")
    return reply


def run_md_or_load_cache(
    mdatoms: MDAtomicData,
    mdconfig: MDConfig,
    server_url: str,
    cache_dir: str,
    label: str,
    endpoint_live: bool,
    timeout: int = 300,
) -> MDReply:
    """Run MD if cache is missing and endpoint is live; otherwise load cache."""
    if cache_exists(cache_dir, label):
        print(f"  Loading cached response: {label}")
        return load_cache(cache_dir, label, MDReply)

    if not endpoint_live:
        raise RuntimeError(
            f"No cached response for '{label}' and endpoint is not available.\n"
            "Start the BMD NIM on localhost:8000 or provide cached_responses/."
        )

    print(f"  Running live MD simulation: {label} ...")
    reply = run_md(mdatoms, mdconfig, server_url, timeout=timeout)
    save_cache(cache_dir, label, reply)
    print(f"  Cached response saved: {label}")
    return reply


def run_bgr_or_load_cache(
    atoms_list: list[AtomicData],
    server_url: str,
    cache_dir: str,
    label: str,
    endpoint_live: bool,
    cellopt: bool = False,
    opttol: float | None = None,
    timeout: int = 300,
) -> BGRReply:
    """Run BGR if cache is missing and endpoint is live; otherwise load cache."""
    if cache_exists(cache_dir, label):
        print(f"  Loading cached response: {label}")
        return load_cache(cache_dir, label, BGRReply)

    if not endpoint_live:
        raise RuntimeError(
            f"No cached response for '{label}' and endpoint is not available.\n"
            "Start the BGR NIM on localhost:8000 or provide cached_responses/."
        )

    print(f"  Running live BGR optimisation: {label} ...")
    reply = run_bgr(atoms_list, server_url, cellopt=cellopt, opttol=opttol, timeout=timeout)
    save_cache(cache_dir, label, reply)
    print(f"  Cached response saved: {label}")
    return reply
