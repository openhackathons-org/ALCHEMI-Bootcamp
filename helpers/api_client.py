"""Notebook-friendly API client for ALCHEMI BMD/BGR endpoints."""

from __future__ import annotations

import asyncio
import math
import warnings

import aiohttp
import requests

from .cache import cache_exists, load_cache, save_cache
from .models import (
    BGRAtomicData,
    BGRReply,
    BGRRequest,
    BMDAtomicData,
    BMDConfig,
    BMDReply,
    BMDRequest,
    BMDSnapshot,
)


def check_endpoint(server_url: str, timeout: int = 5) -> bool:
    """Return True if the ALCHEMI endpoint health-check responds 200."""
    url = f"{server_url}/v2/health/ready"
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.InvalidURL):
        return False


def run_md(
    mdatoms: BMDAtomicData,
    mdconfig: BMDConfig,
    server_url: str,
    timeout: int = 1800,  # 30 minutes
) -> BMDReply:
    """Submit an MD request and return the parsed reply.

    Unlike the CLI script this does *not* loop-wait; it raises on failure.
    """
    url = f"{server_url}/infer"
    payload = BMDRequest(atoms=mdatoms, config=mdconfig).model_dump()
    response = requests.post(url, json=payload, timeout=(10, timeout))
    response.raise_for_status()
    reply = BMDReply(**response.json())
    if reply.status != "Success":
        raise RuntimeError(f"MD simulation failed: {reply.status} — {reply.info}")
    return reply


def run_bgr(
    atoms_list: list[BGRAtomicData],
    server_url: str,
    cellopt: bool = False,
    opttol: float | None = None,
    timeout: int = 1800,  # 30 minutes
) -> BGRReply:
    """Submit a BGR request and return the parsed reply."""
    url = f"{server_url}/infer"
    payload = BGRRequest(
        atoms=atoms_list,
        cellopt=cellopt,
        opttol=opttol,
    ).model_dump()
    response = requests.post(
        url,
        json=payload,
        timeout=(10, timeout),
    )
    response.raise_for_status()
    reply = BGRReply(**response.json())
    if reply.status != "Success":
        raise RuntimeError(f"BGR failed: {reply.status} — {reply.info}")
    return reply


def run_md_or_load_cache(
    mdatoms: BMDAtomicData,
    mdconfig: BMDConfig,
    server_url: str,
    cache_dir: str,
    label: str,
    endpoint_live: bool,
    timeout: int = 1800,  # 30 minutes
) -> BMDReply:
    """Run MD if cache is missing and endpoint is live; otherwise load cache."""
    if cache_exists(cache_dir, label):
        print(f"  Loading cached response: {label}")
        return load_cache(cache_dir, label, BMDReply)

    if not endpoint_live:
        raise RuntimeError(
            f"No cached response for '{label}' and endpoint is not available.\n"
            f"Start the BMD NIM on {server_url} or provide cached_responses/."
        )

    print(f"  Running live MD simulation: {label} ...")
    reply = run_md(mdatoms, mdconfig, server_url, timeout=timeout)
    save_cache(cache_dir, label, reply)
    print(f"  Cached response saved: {label}")
    return reply


def run_bgr_or_load_cache(
    atoms_list: list[BGRAtomicData],
    server_url: str,
    cache_dir: str,
    label: str,
    endpoint_live: bool,
    cellopt: bool = False,
    opttol: float | None = None,
    timeout: int = 1800,  # 30 minutes
) -> BGRReply:
    """Run BGR if cache is missing and endpoint is live; otherwise load cache."""
    if cache_exists(cache_dir, label):
        print(f"  Loading cached response: {label}")
        return load_cache(cache_dir, label, BGRReply)

    if not endpoint_live:
        raise RuntimeError(
            f"No cached response for '{label}' and endpoint is not available.\n"
            f"Start the BGR NIM on {server_url} or provide cached_responses/."
        )

    print(f"  Running live BGR optimisation: {label} ...")
    reply = run_bgr(
        atoms_list,
        server_url,
        cellopt=cellopt,
        opttol=opttol,
        timeout=timeout,
    )
    save_cache(cache_dir, label, reply)
    print(f"  Cached response saved: {label}")
    return reply


# ---------------------------------------------------------------------------
# Async helpers for concurrent temperature sweeps
# ---------------------------------------------------------------------------


def snapshot_to_mdatoms(base: BMDAtomicData, snap: BMDSnapshot) -> BMDAtomicData:
    """Build a new BMDAtomicData from a base structure and a trajectory snapshot."""
    return BMDAtomicData(
        coord=snap.coord,
        numbers=base.numbers,
        charge=base.charge,
        mult=base.mult,
        cell=snap.cell if snap.cell else base.cell,
        pbc=base.pbc,
        velocity=snap.velocity,
    )


async def async_run_md(
    mdatoms: BMDAtomicData,
    mdconfig: BMDConfig,
    server_url: str,
    session: aiohttp.ClientSession,
    timeout: int = 1800,
) -> BMDReply:
    """Async equivalent of ``run_md`` using an aiohttp session."""
    import aiohttp as _aiohttp

    url = f"{server_url}/infer"
    payload = BMDRequest(atoms=mdatoms, config=mdconfig).model_dump()
    # connect timeout must accommodate queueing behind TCPConnector limit
    client_timeout = _aiohttp.ClientTimeout(connect=timeout, total=timeout)
    async with session.post(url, json=payload, timeout=client_timeout) as resp:
        resp.raise_for_status()
        data = await resp.json()
    reply = BMDReply(**data)
    if reply.status != "Success":
        raise RuntimeError(f"MD simulation failed: {reply.status} — {reply.info}")
    return reply


async def async_run_md_or_load_cache(
    mdatoms: BMDAtomicData,
    mdconfig: BMDConfig,
    server_url: str,
    session: aiohttp.ClientSession,
    cache_dir: str,
    label: str,
    endpoint_live: bool,
    timeout: int = 1800,
) -> BMDReply:
    """Async equivalent of ``run_md_or_load_cache``."""
    if cache_exists(cache_dir, label):
        print(f"  Loading cached response: {label}")
        return load_cache(cache_dir, label, BMDReply)

    if not endpoint_live:
        raise RuntimeError(
            f"No cached response for '{label}' and endpoint is not available.\n"
            f"Start the BMD NIM on {server_url} or provide cached_responses/."
        )

    print(f"  Running live MD simulation: {label} ...")
    reply = await async_run_md(mdatoms, mdconfig, server_url, session, timeout=timeout)
    save_cache(cache_dir, label, reply)
    print(f"  Cached response saved: {label}")
    return reply


async def async_run_bgr(
    atoms_list: list[BGRAtomicData],
    server_url: str,
    session: aiohttp.ClientSession,
    cellopt: bool = False,
    opttol: float | None = None,
    timeout: int = 1800,
) -> BGRReply:
    """Async equivalent of ``run_bgr`` using an aiohttp session."""

    url = f"{server_url}/infer"
    payload = BGRRequest(atoms=atoms_list, cellopt=cellopt, opttol=opttol).model_dump()
    # connect timeout must accommodate queueing behind TCPConnector limit
    client_timeout = aiohttp.ClientTimeout(connect=timeout, total=timeout)
    async with session.post(url, json=payload, timeout=client_timeout) as resp:
        resp.raise_for_status()
        data = await resp.json()
    reply = BGRReply(**data)
    if reply.status != "Success":
        raise RuntimeError(f"BGR failed: {reply.status} — {reply.info}")
    return reply


async def async_run_bgr_or_load_cache(
    atoms_list: list[BGRAtomicData],
    server_url: str,
    session: aiohttp.ClientSession,
    cache_dir: str,
    label: str,
    endpoint_live: bool,
    cellopt: bool = False,
    opttol: float | None = None,
    timeout: int = 1800,
) -> BGRReply:
    """Async equivalent of ``run_bgr_or_load_cache``."""
    if cache_exists(cache_dir, label):
        print(f"  Loading cached response: {label}")
        return load_cache(cache_dir, label, BGRReply)

    if not endpoint_live:
        raise RuntimeError(
            f"No cached response for '{label}' and endpoint is not available.\n"
            f"Start the BGR NIM on {server_url} or provide cached_responses/."
        )

    print(f"  Running live BGR optimisation: {label} ...")
    reply = await async_run_bgr(
        atoms_list, server_url, session, cellopt=cellopt, opttol=opttol, timeout=timeout
    )
    save_cache(cache_dir, label, reply)
    print(f"  Cached response saved: {label}")
    return reply


async def async_run_temperature_pipeline(
    temperature: float,
    mdatoms: BMDAtomicData,
    session: aiohttp.ClientSession,
    server_url: str,
    cache_dir: str,
    endpoint_live: bool,
    nvt_time_ps: float,
    npt_time_ps: float,
    dt: float,
    friction: float,
    pressure: float,
    save_interval: int,
    timeout: int,
    semaphore: asyncio.Semaphore,
) -> tuple[float, float, BMDReply | None]:
    """Run NVT→NPT pipeline for a single temperature, return (T, density, npt_reply)."""
    from .analysis import (
        compute_density,
        extract_thermo_timeseries,
        pick_production_window,
    )

    try:
        async with semaphore:
            print(f"\n--- T = {temperature} K ---")

            # NVT equilibration
            nvt_cfg = BMDConfig(
                temperature=float(temperature),
                dt=dt,
                nvt=True,
                npt=False,
                friction=friction,
                md_time_max=nvt_time_ps,
                save_interval=save_interval,
            )
            nvt_reply = await async_run_md_or_load_cache(
                mdatoms,
                nvt_cfg,
                server_url,
                session,
                cache_dir,
                f"nacl_nvt_T{int(temperature)}",
                endpoint_live,
                timeout=timeout,
            )

            # Seed NPT from final NVT snapshot
            seed = snapshot_to_mdatoms(mdatoms, nvt_reply.trajectory[-1])

            # NPT production
            npt_cfg = BMDConfig(
                temperature=float(temperature),
                dt=dt,
                nvt=True,
                npt=True,
                friction=friction,
                pressure=pressure,
                md_time_max=npt_time_ps,
                save_interval=save_interval,
            )
            npt_reply = await async_run_md_or_load_cache(
                seed,
                npt_cfg,
                server_url,
                session,
                cache_dir,
                f"nacl_npt_T{int(temperature)}",
                endpoint_live,
                timeout=timeout,
            )

            # Compute density
            thermo = extract_thermo_timeseries(seed, npt_reply.trajectory)
            s0, s1 = pick_production_window(thermo)
            density = compute_density(seed, thermo["volume_A3"][s0:s1])
            print(f"  T={temperature} K → density = {density:.4f} g/cm³")
            return (temperature, density, npt_reply)

    except Exception as exc:
        warnings.warn(
            f"Pipeline failed for T={temperature} K: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return (temperature, float("nan"), None)


async def async_temperature_sweep(
    temperatures: list[float],
    mdatoms: BMDAtomicData,
    server_url: str,
    cache_dir: str,
    endpoint_live: bool,
    nvt_time_ps: float,
    npt_time_ps: float,
    dt: float,
    friction: float,
    pressure: float,
    save_interval: int,
    timeout: int = 1800,
    max_concurrent: int = 4,
) -> tuple[list[float], list[float], list[BMDReply | None]]:
    """Run NVT→NPT pipelines for all temperatures concurrently.

    Returns ``(temps, densities, npt_replies)`` sorted by temperature.
    Failed pipelines yield NaN density and None reply.
    """
    if not temperatures:
        return ([], [], [])

    import aiohttp as _aiohttp

    semaphore = asyncio.Semaphore(max_concurrent)
    connector = _aiohttp.TCPConnector(limit=max_concurrent)

    async with _aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_run_temperature_pipeline(
                temperature=T,
                mdatoms=mdatoms,
                session=session,
                server_url=server_url,
                cache_dir=cache_dir,
                endpoint_live=endpoint_live,
                nvt_time_ps=nvt_time_ps,
                npt_time_ps=npt_time_ps,
                dt=dt,
                friction=friction,
                pressure=pressure,
                save_interval=save_interval,
                timeout=timeout,
                semaphore=semaphore,
            )
            for T in temperatures
        ]
        results = await asyncio.gather(*tasks)

    # Sort by temperature
    results = sorted(results, key=lambda r: r[0])

    # Warn about failures
    for T, rho, _ in results:
        if math.isnan(rho):
            warnings.warn(
                f"Temperature {T} K returned NaN density (pipeline failed)",
                RuntimeWarning,
                stacklevel=2,
            )

    temps = [r[0] for r in results]
    densities = [r[1] for r in results]
    replies = [r[2] for r in results]
    return (temps, densities, replies)
