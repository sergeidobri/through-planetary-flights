import argparse
import sys

import numpy as np

import calculations as calc
import constants as C
import utils as u


def get_variant(number: int) -> C.Variant:
    """Возвращает вариант задания по номеру с проверкой допустимости."""
    for v in C.VARIANTS:
        if v.number == number:
            return v
    raise ValueError(f"Вариант № {number} не найден. Допустимые номера: "
                     f"{C.VARIANTS[0].number}..{C.VARIANTS[-1].number}")


def run_variant(variant: C.Variant) -> calc.TransferResult:
    """Расчёт траектории перелёта для варианта и печать результатов."""
    print(f"\n>>> Расчёт варианта № {variant.number}: планета {variant.planet}, "
          f"старт {variant.date_start}, t12 = {variant.t12_days:.0f} сут, "
          f"Hисз = {variant.h_isz_km:.0f} км, Hисп = {variant.h_isp_km:.0f} км")
    result = calc.compute_transfer(variant)
    u.print_results(result)
    return result


def build_plots(result: calc.TransferResult, variant: C.Variant,
                out_dir: str, with_scans: bool = True) -> list:
    """Построение и сохранение всех графиков. Возвращает список путей."""
    saved = []

    # График 1: план траектории перелёта в плоскости эклиптики
    saved.append(u.plot_trajectory_2d(result, f"{out_dir}/trajectory_2d.png"))
    print(f"[график] План траектории перелёта: {saved[-1]}")

    # График 2: пространственный вид траектории
    saved.append(u.plot_trajectory_3d(result, f"{out_dir}/trajectory_3d.png"))
    print(f"[график] Пространственная траектория: {saved[-1]}")

    if not with_scans:
        return saved

    # График 3: зависимость затрат от даты старта при фиксированном t12
    date_results = calc.scan_departure_dates(variant)
    jd_base = calc.julian_date_from_str(variant.date_start)
    offsets = np.arange(-C.DATE_SCAN_HALF_RANGE,
                        C.DATE_SCAN_HALF_RANGE + 0.5 * C.DATE_SCAN_STEP,
                        C.DATE_SCAN_STEP)
    saved.append(u.plot_dv_vs_departure_date(
        date_results, jd_base + offsets, variant.t12_days,
        f"{out_dir}/dv_vs_departure_date.png"))
    print(f"[график] Затраты от даты старта: {saved[-1]}")

    # График 4: зависимость затрат от длительности перелёта при фиксированной дате
    time_results = calc.scan_transfer_times(variant)
    times = np.arange(C.TIME_SCAN_MIN,
                      C.TIME_SCAN_MAX + 0.5 * C.TIME_SCAN_STEP,
                      C.TIME_SCAN_STEP)
    saved.append(u.plot_dv_vs_transfer_time(
        time_results, times, variant.date_start,
        f"{out_dir}/dv_vs_transfer_time.png"))
    print(f"[график] Затраты от длительности перелёта: {saved[-1]}")

    # График 5: изолинии суммарных затрат dV_сум(D1, t12)
    dates_jd, t_grid, dv_grid = calc.scan_grid(variant)
    saved.append(u.plot_dv_contour(dates_jd, t_grid, dv_grid,
                                   f"{out_dir}/dv_contour.png"))
    print(f"[график] Изолинии суммарных затрат: {saved[-1]}")

    # Краткий анализ зависимостей: минимумы в просканированных диапазонах
    _print_scan_analysis(date_results, offsets, time_results, times)
    return saved


def _print_scan_analysis(date_results, offsets, time_results, times) -> None:
    """Вывод простого анализа полученных зависимостей энергетических затрат."""
    valid_d = [(r, o) for r, o in zip(date_results, offsets) if r is not None]
    valid_t = [(r, t) for r, t in zip(time_results, times) if r is not None]
    print("\n--- Анализ зависимостей энергетических затрат ".ljust(78, "-"))
    if valid_d:
        r_min, o_min = min(valid_d, key=lambda p: p[0].dV_total)
        d, m, y = calc.jd_to_date(calc.julian_date_from_str(r_min.variant.date_start))
        print(f"Минимум dV_сум по дате старта:   {r_min.dV_total:.3f} км/с "
              f"при D1 = {d:02d}.{m:02d}.{y:04d} (t12 = {r_min.variant.t12_days:.0f} сут)")
    if valid_t:
        r_min, t_min = min(valid_t, key=lambda p: p[0].dV_total)
        print(f"Минимум dV_сум по длительности:  {r_min.dV_total:.3f} км/с "
              f"при t12 = {t_min:.0f} сут (D1 = {r_min.variant.date_start})")


def parse_args(argv=None) -> argparse.Namespace:
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Расчет межпланетных траекторий перелёта Земля - Марс ")
    parser.add_argument("--variant", type=int, default=C.DEFAULT_VARIANT_NUMBER,
                        help="номер варианта задания (1..34), по умолчанию "
                             f"{C.DEFAULT_VARIANT_NUMBER}")
    parser.add_argument("--outdir", type=str, default=C.RESULTS_DIR,
                        help="каталог для результатов (по умолчанию results/)")
    parser.add_argument("--no-scans", action="store_true",
                        help="не выполнять сканирование и не строить "
                             "зависимости энергетических затрат")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Основная логика программы: расчёт варианта, отчёт, графики."""
    u.setup_russian_font()                      # русские подписи графиков
    args = parse_args(argv)

    try:
        variant = get_variant(args.variant)
        out_dir = u.ensure_dir(args.outdir)

        result = run_variant(variant)

        report_path = f"{out_dir}/report_variant_{variant.number:02d}.txt"
        u.write_report(result, report_path)
        print(f"\n[отчёт] Текстовый отчёт сохранён: {report_path}")

        build_plots(result, variant, out_dir, with_scans=not args.no_scans)

        print("\nРасчёт завершён успешно.")
        return 0
    except ValueError as exc:
        print(f"ОШИБКА ИСХОДНЫХ ДАННЫХ: {exc}", file=sys.stderr)
        return 1
    except calc.CalculationError as exc:
        print(f"ОШИБКА РАСЧЁТА: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ОШИБКА ЗАПИСИ ФАЙЛОВ: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
