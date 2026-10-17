"""Ejecuta de principio a fin el pipeline final de datos."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STAGES = (
    (1, "Armonizar exportaciones nuevas (PT2)", "01_harmonize_new_exports.py"),
    (2, "Armonizar y fusionar exportaciones PT3", "01b_harmonize_pt3_exports.py"),
    (3, "Construir dataset raw final", "02_build_final_raw_dataset.py"),
    (4, "Limpiar dataset final", "03_clean_final_dataset.py"),
    (5, "Construir features finales", "04_build_final_features.py"),
    (6, "Validar outputs finales", "05_validate_final_outputs.py"),
    (7, "Generar EDA final", "06_run_final_eda.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta todas o parte de las etapas del pipeline final."
    )
    parser.add_argument(
        "--start-at",
        type=int,
        choices=range(1, len(STAGES) + 1),
        default=1,
        metavar="ETAPA",
        help="Primera etapa que se ejecutará (1-7).",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        choices=range(1, len(STAGES) + 1),
        default=len(STAGES),
        metavar="ETAPA",
        help="Última etapa que se ejecutará (1-7).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Muestra las etapas y termina sin ejecutar el pipeline.",
    )
    return parser.parse_args()


def print_stages() -> None:
    print("Etapas del pipeline final:")
    for number, description, script_name in STAGES:
        print(f"  {number}. {description} ({script_name})")


def validate_stage_range(start_at: int, stop_after: int) -> None:
    if start_at > stop_after:
        raise ValueError("--start-at no puede ser mayor que --stop-after")

    missing = [name for _, _, name in STAGES if not (SCRIPTS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan scripts del pipeline: {', '.join(missing)}")


def run_stage(number: int, description: str, script_name: str) -> float:
    script_path = SCRIPTS_DIR / script_name
    started_at = time.monotonic()

    print("\n" + "=" * 72, flush=True)
    print(f"ETAPA {number}/{len(STAGES)}: {description}", flush=True)
    print(f"Script: {script_name}", flush=True)
    print("=" * 72, flush=True)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    elapsed = time.monotonic() - started_at
    print(f"ETAPA {number} COMPLETADA EN {elapsed:.1f} s", flush=True)
    return elapsed


def main() -> int:
    args = parse_args()

    if args.list:
        print_stages()
        return 0

    try:
        validate_stage_range(args.start_at, args.stop_after)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    selected = [
        stage for stage in STAGES if args.start_at <= stage[0] <= args.stop_after
    ]
    pipeline_started_at = time.monotonic()

    print(f"Pipeline final: etapas {args.start_at} a {args.stop_after}")
    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")

    try:
        for stage in selected:
            run_stage(*stage)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nPIPELINE DETENIDO: la etapa {stage[0]} falló "
            f"con código {exc.returncode}.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except KeyboardInterrupt:
        print("\nPIPELINE INTERRUMPIDO POR EL USUARIO.", file=sys.stderr)
        return 130

    elapsed = time.monotonic() - pipeline_started_at
    print("\n" + "=" * 72)
    print(f"PIPELINE FINAL COMPLETADO EN {elapsed:.1f} s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
