import datetime as _dt
import os
from typing import List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # работа без графического окна
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

import calculations as calc
import constants as C

_RU_FONT_CONFIGURED = False

def setup_russian_font() -> None:
    """
    Однократная настройка matplotlib: семейство шрифтов с поддержкой кириллицы.
    """
    global _RU_FONT_CONFIGURED
    if _RU_FONT_CONFIGURED:
        return
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for candidate in ("DejaVu Sans", "Arial", "Liberation Sans", "Sans"):
        if candidate in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = C.PLOT_GRID_ALPHA
    plt.rcParams["figure.dpi"] = 100
    _RU_FONT_CONFIGURED = True

# форматированный вывод
def fmt_deg(rad: float, digits: int = 3) -> str:
    """Форматирование угла в градусах: '123.456°'."""
    return f"{rad * C.DEG:.{digits}f}°"


def fmt_vec(v: Sequence[float], digits: int = 4) -> str:
    """Форматирование вектора: '(x; y; z)'."""
    return "(" + "; ".join(f"{x:.{digits}f}" for x in v) + ")"


def ensure_dir(path: str) -> str:
    """Создаёт каталог, если он отсутствует. Возвращает путь."""
    os.makedirs(path, exist_ok=True)
    return path


def _line(title: str) -> str:
    return f"--- {title} ".ljust(78, "-")


def build_report_lines(res: calc.TransferResult) -> List[str]:
    """Формирует строки текстового отчёта по результату расчёта."""
    v = res.variant
    d1 = calc.jd_to_date(res.D1)
    d2 = calc.jd_to_date(res.D2)
    lines = [
        "=" * 78,
        " РАСЧЁТ МЕЖПЛАНЕТНОЙ ТРАЕКТОРИИ ПЕРЁЛЕТА ЗЕМЛЯ - МАРС",
        "=" * 78,
        _line("ИСХОДНЫЕ ДАННЫЕ ВАРИАНТА"),
        f"Номер варианта:            {v.number}",
        f"Планета назначения:        {v.planet}",
        f"Дата старта D1:            {v.date_start}  (JD = {res.D1:.1f})",
        f"Время перелёта t12:        {v.t12_days:.0f} сут",
        f"Дата прилёта D2:           {d2[0]:02d}.{d2[1]:02d}.{d2[2]:04d}  (JD = {res.D2:.1f})",
        f"Высота орбиты ИСЗ:         {v.h_isz_km:.0f} км",
        f"Высота орбиты ИС Марса:    {v.h_isp_km:.0f} км",
        _line("ПОЛОЖЕНИЕ ПЛАНЕТ НА ГЕЛИОЦЕНТРИЧЕСКОЙ ОРБИТЕ (п. 2.1)"),
        f"Земля,  D1: r1 = {fmt_vec(res.state1.r_vec / 1e6)} млн км, |r1| = {res.state1.r / 1e6:.6f} млн км",
        f"        V1 = {fmt_vec(res.state1.v_vec)} км/с,  тета1(орб. Земли) = {fmt_deg(res.state1.theta)}",
        f"Марс,   D2: r2 = {fmt_vec(res.state2.r_vec / 1e6)} млн км, |r2| = {res.state2.r / 1e6:.6f} млн км",
        f"        V2 = {fmt_vec(res.state2.v_vec)} км/с,  тета2(орб. Марса) = {fmt_deg(res.state2.theta)}",
        _line("ПАРАМЕТРЫ ГЕЛИОЦЕНТРИЧЕСКОГО УЧАСТКА (внешняя задача)"),
        f"Наклонение плоскости i_т.п.:      {fmt_deg(res.i_tp)}",
        f"Долгота восх. узла Омега_т.п.:    {fmt_deg(res.omega_node_tp)}",
        f"Угловая дальность перелёта фи:    {fmt_deg(res.phi)}",
        f"Длина хорды s:                    {res.chord_s / 1e6:.6f} млн км",
        f"Время по граничной орбите t_гр:   {res.t_boundary / C.SEC_PER_DAY:.2f} сут",
        f"Род орбиты перелёта:              {res.orbit_kind} "
        f"({'t12 < t_гр' if res.orbit_kind == 1 else 't12 > t_гр'})",
        f"Большая полуось a_т.п.:           {res.a_tp / 1e6:.6f} млн км "
        f"({res.a_tp / C.AU:.6f} а.е.)",
        f"Эксцентриситет e_т.п.:            {res.e_tp:.6f}",
        f"Аргумент перицентра омега_т.п.:   {fmt_deg(res.omega_tp)}",
        f"Истинная аномалия старта тета1:   {fmt_deg(res.theta1_tp)}",
        f"Истинная аномалия прилёта тета2:  {fmt_deg(res.theta2_tp)}",
        _line("СКОРОСТИ КА И ИМПУЛЬСЫ НА ГРАНИЦАХ СФЕР ДЕЙСТВИЯ (п. 2.3)"),
        f"V_КА1 = {fmt_vec(res.v_sc1)} км/с",
        f"V_КА2 = {fmt_vec(res.v_sc2)} км/с",
        f"dV1 (вектор) = {fmt_vec(res.dV1_vec)} км/с,  |dV1| = {res.dV1:.4f} км/с",
        f"dV2 (вектор) = {fmt_vec(res.dV2_vec)} км/с,  |dV2| = {res.dV2:.4f} км/с",
        _line("ВНУТРЕННЯЯ ЗАДАЧА: ЭНЕРГЕТИЧЕСКИЕ ЗАТРАТЫ (п. 3)"),
        f"Гиперболический избыток у Земли:  V_сф.з = {res.dV1:.4f} км/с",
        f"Гиперболический избыток у Марса:  V_сф.п = {res.dV2:.4f} км/с",
        f"Импульс ухода с орбиты ИСЗ:       dV_исз = {res.dV_isz:.4f} км/с",
        f"Импульс выхода на орбиту ИС Марса: dV_исп = {res.dV_isp:.4f} км/с",
        f"СУММАРНЫЕ ЗАТРАТЫ:                dV_сум = {res.dV_total:.4f} км/с",
        _line("КОНТРОЛЬ ТОЧНОСТИ"),
        f"Невязка прохождения точки r1:     {res.check_residual_km:.3e} км",
        "=" * 78,
    ]
    return lines


def print_results(res: calc.TransferResult) -> None:
    """Печать таблицы результатов в консоль."""
    print("\n".join(build_report_lines(res)))


def write_report(res: calc.TransferResult, path: str) -> None:
    """Сохранение текстового отчёта в файл."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(build_report_lines(res)) + "\n")


# утилиты
def planet_orbit_points(elements: C.OrbitalElements, n: int = 400) -> np.ndarray:
    """Точки полной орбиты планеты в гелиоцентрической системе OXYZ, км."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    r = elements.a * (1.0 - elements.e ** 2) / (1.0 + elements.e * np.cos(theta))
    r_orb = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
    omega = elements.pi_lon - elements.omega_node
    A = calc.rotation_matrix(elements.i, elements.omega_node, omega)
    return A @ r_orb


def transfer_arc_points(res: calc.TransferResult, n: int = 400) -> np.ndarray:
    """Точки дуги траектории перелёта от тета1 до тета1 + фи, км."""
    theta = np.linspace(res.theta1_tp, res.theta1_tp + res.phi, n)
    p = res.a_tp * (1.0 - res.e_tp ** 2)
    r = p / (1.0 + res.e_tp * np.cos(theta))
    r_orb = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
    A = calc.rotation_matrix(res.i_tp, res.omega_node_tp, res.omega_tp)
    return A @ r_orb


def _save(fig, path: str) -> str:
    """Сохранение рисунка в файл с обработкой ошибок записи."""
    fig.tight_layout()
    fig.savefig(path, dpi=C.PLOT_DPI)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# График 1: траектория перелёта в плоскости эклиптики (2D)
# ----------------------------------------------------------------------------
def plot_trajectory_2d(res: calc.TransferResult, path: str) -> str:
    """План траектории перелёта: орбиты планет, дуга перелёта, точки старта/прилёта."""
    setup_russian_font()
    fig, ax = plt.subplots(figsize=(9, 8))

    earth = planet_orbit_points(C.EARTH) / 1e6
    mars = planet_orbit_points(C.PLANETS[res.variant.planet]) / 1e6
    arc = transfer_arc_points(res) / 1e6

    ax.plot(earth[0], earth[1], color="tab:blue", lw=C.PLOT_LINE_WIDTH,
            label="Орбита Земли")
    ax.plot(mars[0], mars[1], color="tab:red", lw=C.PLOT_LINE_WIDTH,
            label=f"Орбита {res.variant.planet}а")
    ax.plot(arc[0], arc[1], color="tab:green", lw=C.PLOT_LINE_WIDTH + 0.4,
            label="Траектория перелёта")
    ax.plot([0.0], [0.0], marker="o", ms=14, color="gold",
            markeredgecolor="orange", label="Солнце")
    ax.plot([res.state1.r_vec[0] / 1e6], [res.state1.r_vec[1] / 1e6],
            marker="o", ms=8, color="tab:blue", markeredgecolor="k",
            label=f"Точка отлёта D1 ({res.variant.date_start})")
    ax.plot([res.state2.r_vec[0] / 1e6], [res.state2.r_vec[1] / 1e6],
            marker="s", ms=8, color="tab:red", markeredgecolor="k",
            label="Точка прилёта D2")

    ax.set_xlabel("X, млн км")
    ax.set_ylabel("Y, млн км")
    ax.set_title("Гелиоцентрическая траектория перелёта Земля - "
                 f"{res.variant.planet} (проекция на плоскость эклиптики)\n"
                 f"Вариант {res.variant.number}: t12 = {res.variant.t12_days:.0f} сут")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    return _save(fig, path)


# ----------------------------------------------------------------------------
# График 2: трёхмерное представление траектории перелёта
# ----------------------------------------------------------------------------
def plot_trajectory_3d(res: calc.TransferResult, path: str) -> str:
    """Трёхмерный вид траектории перелёта в эклиптической системе OXYZ."""
    setup_russian_font()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    earth = planet_orbit_points(C.EARTH) / 1e6
    mars = planet_orbit_points(C.PLANETS[res.variant.planet]) / 1e6
    arc = transfer_arc_points(res) / 1e6

    ax.plot(earth[0], earth[1], earth[2], color="tab:blue",
            lw=C.PLOT_LINE_WIDTH, label="Орбита Земли")
    ax.plot(mars[0], mars[1], mars[2], color="tab:red",
            lw=C.PLOT_LINE_WIDTH, label=f"Орбита {res.variant.planet}а")
    ax.plot(arc[0], arc[1], arc[2], color="tab:green",
            lw=C.PLOT_LINE_WIDTH + 0.4, label="Траектория перелёта")
    ax.scatter([0], [0], [0], s=150, color="gold", edgecolors="orange",
               label="Солнце")
    ax.scatter(*(res.state1.r_vec / 1e6), s=60, color="tab:blue",
               edgecolors="k", label="Точка отлёта D1")
    ax.scatter(*(res.state2.r_vec / 1e6), s=60, color="tab:red",
               edgecolors="k", label="Точка прилёта D2")

    ax.set_xlabel("X, млн км")
    ax.set_ylabel("Y, млн км")
    ax.set_zlabel("Z, млн км")
    ax.set_title("Пространственная траектория перелёта Земля - "
                 f"{res.variant.planet}\n"
                 f"i_т.п. = {fmt_deg(res.i_tp)}, Омега_т.п. = {fmt_deg(res.omega_node_tp)}")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


# ----------------------------------------------------------------------------
# График 3/4: зависимости энергетических затрат
# ----------------------------------------------------------------------------
def _dv_curves(results: Sequence[Optional[calc.TransferResult]]):
    """Извлекает из списка результатов массивы dV_исз, dV_исп, dV_сум (km/s)."""
    arr = np.array([[np.nan, np.nan, np.nan] if r is None
                    else [r.dV_isz, r.dV_isp, r.dV_total] for r in results])
    return arr[:, 0], arr[:, 1], arr[:, 2]


def plot_dv_vs_departure_date(results: Sequence[Optional[calc.TransferResult]],
                              jd_dates: np.ndarray, t12: float, path: str) -> str:
    """Зависимость характеристических затрат от даты старта (t12 фиксировано)."""
    setup_russian_font()
    dv_isz, dv_isp, dv_sum = _dv_curves(results)
    dates = [calc.jd_to_date(jd) for jd in jd_dates]
    x = [_dt.date(d[2], d[1], d[0]) for d in dates]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, dv_isz, "o-", ms=3, lw=C.PLOT_LINE_WIDTH, color="tab:blue",
            label="ΔV исз — уход с орбиты ИСЗ")
    ax.plot(x, dv_isp, "s-", ms=3, lw=C.PLOT_LINE_WIDTH, color="tab:red",
            label="ΔV исп — выход на орбиту ИС Марса")
    ax.plot(x, dv_sum, "^-", ms=3, lw=C.PLOT_LINE_WIDTH, color="tab:green",
            label="ΔV сум — суммарные затраты")

    ax.set_xlabel("Дата старта D1")
    ax.set_ylabel("Характеристическая скорость, км/с")
    ax.set_title(f"Зависимость энергетических затрат от даты старта\n"
                 f"(перелёт Земля - Марс, t12 = {t12:.0f} сут)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
    fig.autofmt_xdate()
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_dv_vs_transfer_time(results: Sequence[Optional[calc.TransferResult]],
                             times: np.ndarray, date_start: str, path: str) -> str:
    """Зависимость характеристических затрат от длительности перелёта."""
    setup_russian_font()
    dv_isz, dv_isp, dv_sum = _dv_curves(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, dv_isz, "o-", ms=3, lw=C.PLOT_LINE_WIDTH, color="tab:blue",
            label="ΔV исз — уход с орбиты ИСЗ")
    ax.plot(times, dv_isp, "s-", ms=3, lw=C.PLOT_LINE_WIDTH, color="tab:red",
            label="ΔV исп — выход на орбиту ИС Марса")
    ax.plot(times, dv_sum, "^-", ms=3, lw=C.PLOT_LINE_WIDTH, color="tab:green",
            label="ΔV сум — суммарные затраты")

    ax.set_xlabel("Длительность перелёта t12, сут")
    ax.set_ylabel("Характеристическая скорость, км/с")
    ax.set_title(f"Зависимость энергетических затрат от длительности перелёта\n"
                 f"(перелёт Земля - Марс, дата старта {date_start})")
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_dv_contour(dates_jd: np.ndarray, times: np.ndarray, dv: np.ndarray,
                    path: str) -> str:
    """Изолинии суммарных характеристических затрат dV_сум(дата старта, t12)."""
    setup_russian_font()
    x = mdates.date2num([_dt.date(*reversed(calc.jd_to_date(jd))) for jd in dates_jd])
    X, Y = np.meshgrid(x, times)
    masked = np.ma.masked_invalid(dv)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    cs = ax.contourf(X, Y, masked, levels=20, cmap="viridis")
    lines = ax.contour(X, Y, masked, levels=10, colors="white", linewidths=0.6)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.1f")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Суммарные затраты ΔV сум, км/с")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
    fig.autofmt_xdate()
    ax.set_xlabel("Дата старта D1")
    ax.set_ylabel("Длительность перелёта t12, сут")
    ax.set_title("Изолинии суммарных характеристических затрат перелёта Земля - Марс")
    return _save(fig, path)


if __name__ == "__main__":
    # Пример использования: настройка русских шрифтов и форматирование вывода
    setup_russian_font()
    print("Настроен семейство шрифтов:", plt.rcParams["font.sans-serif"])
    print("Пример форматирования угла:", fmt_deg(3.14159 / 3))
    print("Пример форматирования вектора:", fmt_vec([1.5, -2.25, 0.125]))
