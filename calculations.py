from dataclasses import dataclass, replace
from typing import List, Tuple
import datetime as _dt

import numpy as np

import constants as C


# кастомные исключения
class CalculationError(RuntimeError):
    """Ошибка баллистического расчёта (расходимость итераций и т.п.)."""


# структуры данных
@dataclass
class StateVector:
    """Фазовые координаты точки орбиты в гелиоцентрической системе OXYZ."""
    r_vec: np.ndarray         # радиус-вектор, км
    v_vec: np.ndarray         # вектор скорости, км/с
    r: float                  # модуль радиус-вектора, км
    theta: float              # истинная аномалия, рад
    E: float                  # эксцентрическая аномалия, рад
    M: float                  # средняя аномалия, рад


@dataclass
class TransferResult:
    """Полный результат расчёта межпланетной траектории перелёта."""
    variant: C.Variant
    D1: float                 # юлианская дата старта
    D2: float                 # юлианская дата прилёта
    state1: StateVector       # состояние Земли в момент D1
    state2: StateVector       # состояние планеты назначения в момент D2
    i_tp: float               # наклонение плоскости перелёта, рад
    omega_node_tp: float      # долгота восходящего узла плоскости перелёта, рад
    phi: float                # угловая дальность перелёта, рад
    chord_s: float            # длина хорды, км
    t_boundary: float         # время перелёта по граничной орбите, с
    orbit_kind: int           # род орбиты перелёта (1 или 2)
    a_tp: float               # большая полуось орбиты перелёта, км
    e_tp: float               # эксцентриситет орбиты перелёта
    omega_tp: float           # аргумент перицентра орбиты перелёта, рад
    theta1_tp: float          # истинная аномалия точки старта на т.п., рад
    theta2_tp: float          # истинная аномалия точки прилёта на т.п., рад
    v_sc1: np.ndarray         # скорость КА в точке старта (гелиоцентр.), км/с
    v_sc2: np.ndarray         # скорость КА в точке прилёта (гелиоцентр.), км/с
    dV1_vec: np.ndarray       # импульс ухода от Земли (вектор), км/с
    dV2_vec: np.ndarray       # импульс прилёта (вектор), км/с
    dV1: float                # модуль импульса ухода от Земли, км/с
    dV2: float                # модуль импульса прилёта, км/с
    dV_isz: float             # импульс ухода с орбиты ИСЗ, км/с
    dV_isp: float             # импульс выхода на орбиту ИС планеты, км/с
    dV_total: float           # суммарные характеристические затраты, км/с
    check_residual_km: float  # невязка контроля прохождения точки r1, км


# работа с датами
def parse_date(date_str: str) -> Tuple[int, int, int]:
    """Разбирает строку даты формата ДД.ММ.ГГГГ в кортеж (день, месяц, год)."""
    parts = date_str.strip().split(".")
    if len(parts) != 3:
        raise ValueError(f"Неверный формат даты: '{date_str}' (ожидается ДД.ММ.ГГГГ)")
    try:
        day, month, year = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"Неверный формат даты: '{date_str}'") from exc
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"Некорректные значения дня/месяца в дате '{date_str}'")
    try:                                
        _dt.date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"Некорректная дата: '{date_str}' ({exc})") from exc
    return day, month, year


def julian_date(day: int, month: int, year: int) -> float:
    """
    Юлианская дата D для даты григорианского календаря:
    целые сутки отсчитываются от полуночи (D = JD_0h - 0.5), поэтому эпоха
    1 января 1957 г. соответствует D0 = 2435839.
    """
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    jd0 = (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1))
           + day + b - 1524.5)          # JD на 00:00 UT
    return jd0 - 0.5


def julian_date_from_str(date_str: str) -> float:
    """Юлианская дата для строки ДД.ММ.ГГГГ."""
    day, month, year = parse_date(date_str)
    return julian_date(day, month, year)


def jd_to_date(jd: float) -> Tuple[int, int, int]:
    """
    Обратное преобразование: из юлианской даты D (соглашение: целые сутки от
    полуночи) в календарные (день, месяц, год).
    """
    z = int(np.floor(jd + 1.0))
    alpha = int((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return day, month, year


# уравнение Кеплера
def solve_kepler(M: float, e: float,
                 tol: float = C.KEPLER_TOL,
                 max_iter: int = C.KEPLER_MAX_ITER) -> float:
    """
    Решение уравнения Кеплера E - e*sin(E) = M методом простых итераций (1).
    Возвращает эксцентрическую аномалию E, рад.
    """
    M = (M + C.PI) % (2.0 * C.PI) - C.PI
    E = M if e < 0.8 else C.PI
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            return E
    raise CalculationError(f"Уравнение Кеплера не сошлось (M={M:.6f}, e={e})")


def rotation_matrix(i: float, omega_node: float, omega: float) -> np.ndarray:
    """
    Матрица перехода A от орбитальной системы координат к гелиоцентрической
    эклиптической OXYZ, формулы (8)-(9).
    """
    ci, si = np.cos(i), np.sin(i)
    cO, sO = np.cos(omega_node), np.sin(omega_node)
    cw, sw = np.cos(omega), np.sin(omega)

    return np.array([
        [cw * cO - sw * sO * ci, -sw * cO - cw * sO * ci,  sO * si],
        [cw * sO + sw * cO * ci, -sw * sO + cw * cO * ci, -cO * si],  # ошибка методички
        [sw * si,                cw * si,                ci],
    ])


def planet_state(elements: C.OrbitalElements, D: float) -> StateVector:
    """
    Фазовые координаты планеты на юлианскую дату D:
    уравнения Кеплера (1), истинная аномалия (2), радиус (3),
    составляющие скорости (4), переход к OXYZ (5)-(9).
    """
    # Средняя аномалия на дату D: M = M0 + n*(D - D0), M0 = lambda0 - pi0
    M0 = elements.lam - elements.pi_lon
    M = M0 + elements.n * (D - C.D0)
    E = solve_kepler(M, elements.e)

    # Истинная аномалия, формула (2)
    theta = 2.0 * np.arctan2(np.sqrt(1.0 + elements.e) * np.sin(E / 2.0),
                             np.sqrt(1.0 - elements.e) * np.cos(E / 2.0))
    theta %= 2.0 * C.PI

    # Радиус-вектор, формула (3)
    r = elements.a * (1.0 - elements.e ** 2) / (1.0 + elements.e * np.cos(theta))

    # Радиальная и трансверсальная составляющие скорости, формула (4).
    # Движение планеты вокруг Солнца: гравитационный параметр MU_SUN.
    p_fac = np.sqrt(C.MU_SUN / (elements.a * (1.0 - elements.e ** 2)))
    v_r = p_fac * elements.e * np.sin(theta)
    v_tau = p_fac * (1.0 + elements.e * np.cos(theta))

    # Координаты и скорость в орбитальной системе, формулы (5)-(6)
    r_orb = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
    v_orb = np.array([v_r * np.cos(theta) - v_tau * np.sin(theta),
                      v_r * np.sin(theta) + v_tau * np.cos(theta),
                      0.0])

    # Переход к гелиоцентрической системе, формулы (7)-(9)
    omega = elements.pi_lon - elements.omega_node       # омега = пи - Омега
    A = rotation_matrix(elements.i, elements.omega_node, omega)
    r_vec = A @ r_orb
    v_vec = A @ v_orb
    return StateVector(r_vec, v_vec, r, theta, E, M)


# параметры плоскости траектории
def transfer_plane(r1: np.ndarray, r2: np.ndarray) -> Tuple[float, float]:
    """
    Наклонение i_т.п. и долгота восходящего узла Омега_т.п. плоскости
    гелиоцентрического участка, формулы (10)-(11).
    """
    c = np.cross(r1, r2)                    # вектор r1 x r2
    c_mag = np.linalg.norm(c)
    if c_mag < C.ANGLE_TOL:
        raise CalculationError("Векторы r1 и r2 коллинеарны: плоскость перелёта не определена")
    sgn = 1.0 if c[2] >= 0.0 else -1.0       # sign(x1*y2 - x2*y1)

    # Формула (10): 0 < i_т.п. < pi/2
    cos_i = abs(c[2]) / c_mag
    i_tp = np.arccos(np.clip(cos_i, -1.0, 1.0))
    if np.sin(i_tp) < C.ANGLE_TOL:
        raise CalculationError("Плоскость перелёта совпадает с плоскостью эклиптики: Омега не определена")

    # Формула (11)
    sin_O = c[0] * sgn / (np.sin(i_tp) * c_mag)
    cos_O = (r1[0] * r2[2] - r2[0] * r1[2]) * sgn / (np.sin(i_tp) * c_mag)
    omega_node_tp = np.arctan2(sin_O, cos_O) % (2.0 * C.PI)
    return i_tp, omega_node_tp


def angular_range(r1: np.ndarray, r2: np.ndarray) -> float:
    """Угловая дальность перелёта фи, формула (12), 0 < фи < 2*pi."""
    c = np.cross(r1, r2)
    sgn = 1.0 if c[2] >= 0.0 else -1.0
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    cos_phi = np.dot(r1, r2) / (n1 * n2)
    sin_phi = np.linalg.norm(c) * sgn / (n1 * n2)
    return np.arctan2(sin_phi, cos_phi) % (2.0 * C.PI)


# Уравнение Ламберта
def boundary_transfer_time(r1: float, r2: float, chord_s: float, phi: float) -> Tuple[float, float]:
    """
    Время перелёта по граничной эллиптической орбите t_гр и её большая полуось,
    формула (13). Знак «+» при pi < фи < 2*pi, знак «-» при 0 < фи < pi.
    """
    a_gr = 0.25 * (r1 + r2 + chord_s)
    beta_g = 2.0 * np.arcsin(np.sqrt((r1 + r2 - chord_s) / (r1 + r2 + chord_s)))
    sign = 1.0 if phi > C.PI else -1.0
    t_gr = (a_gr ** 1.5 / np.sqrt(C.MU_SUN)) * (C.PI + sign * (beta_g - np.sin(beta_g)))
    return t_gr, a_gr


def lambert_time_of_flight(a: float, r1: float, r2: float, chord_s: float,
                           phi: float, orbit_kind: int) -> float:
    """
    Время перелёта по эллиптической орбите с большой полуосью a:
    формула (14) для орбиты первого рода, формула (16) — второго рода.
    """
    z_plus = np.sqrt((r1 + r2 + chord_s) / (4.0 * a))
    z_minus = np.sqrt((r1 + r2 - chord_s) / (4.0 * a))
    alpha = 2.0 * np.arcsin(z_plus) - np.sin(2.0 * np.arcsin(z_plus))
    beta = 2.0 * np.arcsin(z_minus) - np.sin(2.0 * np.arcsin(z_minus))
    if orbit_kind == 2:
        # Орбита второго рода: pi/2 < eps/2 < pi  (ограничения (19))
        alpha = 2.0 * C.PI - alpha
    sign = 1.0 if phi > C.PI else -1.0
    return (a ** 1.5 / np.sqrt(C.MU_SUN)) * (alpha + sign * beta)


def solve_lambert_semimajor(r1: float, r2: float, chord_s: float,
                            t12_sec: float, phi: float) -> Tuple[float, int]:
    """
    Решение уравнения Ламберта: определение большой полуоси a орбиты перелёта
    по заданному времени t12. Возвращает (a, род орбиты).
    """
    t_gr, a_gr = boundary_transfer_time(r1, r2, chord_s, phi)

    if abs(t12_sec - t_gr) <= C.LAMBERT_TOL * t_gr:
        return a_gr, 1                       # вырожденный случай: a = a_гр

    orbit_kind = 1 if t12_sec < t_gr else 2

    def f(a: float) -> float:
        return lambert_time_of_flight(a, r1, r2, chord_s, phi, orbit_kind) - t12_sec

    # Построение начального отрезка бисекции (функция монотонна по a)
    lo, hi = a_gr, a_gr * 2.0
    f_lo = f(lo)
    for _ in range(200):
        f_hi = f(hi)
        if f_lo * f_hi <= 0.0:
            break
        hi *= 2.0
    else:
        raise CalculationError("Не удалось построить отрезок бисекции в уравнении Ламберта")

    # Метод бисекции
    for _ in range(C.LAMBERT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(hi - lo) <= C.LAMBERT_TOL * mid:
            return mid, orbit_kind
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    raise CalculationError("Уравнение Ламберта не сошлось за число итераций")


def eccentricity_and_anomalies(a: float, r1: float, r2: float, chord_s: float,
                               phi: float, orbit_kind: int
                               ) -> Tuple[float, float, float, float, float]:
    """
    Эксцентриситет орбиты перелёта (17) с ограничениями (18)-(19) и
    эксцентрические аномалии точек 1 и 2 (20).
    Возвращает (e, eps, delta, E1, E2).
    """
    x_eps = np.arcsin(np.sqrt((r1 + r2 + chord_s) / (4.0 * a)))
    x_del = np.arcsin(np.sqrt((r1 + r2 - chord_s) / (4.0 * a)))

    # Выбор ветвей согласно ограничениям (18)-(19)
    eps_half = x_eps if orbit_kind == 1 else C.PI - x_eps
    del_half = x_del if phi < C.PI else 2.0 * C.PI - x_del
    eps, delta = 2.0 * eps_half, 2.0 * del_half

    # (eps - delta)/2 = (E2 - E1)/2 для прямого перелёта лежит в (0, pi);
    # приведение по модулю 2*pi необходимо при фи > pi (ветви delta/2 > 3*pi/2)
    half_diff = (eps_half - del_half) % (2.0 * C.PI)
    cos_hd, sin_hd = np.cos(half_diff), np.sin(half_diff)
    if abs(cos_hd) < C.ANGLE_TOL or abs(sin_hd) < C.ANGLE_TOL:
        raise CalculationError("Вырожденная геометрия в формуле эксцентриситета (17)")

    # Формула (17)
    term1 = (2.0 * a - r1 - r2) / (2.0 * a * cos_hd)
    term2 = (r2 - r1) / (2.0 * a * sin_hd)
    e = np.hypot(term1, term2)

    # Формулы (20): синус и косинус (E1 + E2)/2
    sin_sum = (r2 - r1) / (2.0 * a * e * sin_hd)
    cos_sum = (2.0 * a - r1 - r2) / (2.0 * a * e * cos_hd)
    sum_half = np.arctan2(sin_sum, cos_sum)  # (E1 + E2)/2

    E1 = sum_half - half_diff                # E2 - E1 = eps - delta
    E2 = sum_half + half_diff
    return e, eps, delta, E1, E2


def true_anomaly_from_E(E: float, e: float) -> float:
    """Истинная аномалия по эксцентрической, формула (21)."""
    theta = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                             np.sqrt(1.0 - e) * np.cos(E / 2.0))
    return theta % (2.0 * C.PI)


def latitude_argument(r: np.ndarray, omega_node: float) -> float:
    """Аргумент широты точки u, формула (23), 0 < u < 2*pi."""
    n = np.linalg.norm(r)
    cos_u = (r[0] * np.cos(omega_node) + r[1] * np.sin(omega_node)) / n
    aux = r[1] * np.cos(omega_node) - r[0] * np.sin(omega_node)
    sin_u = np.hypot(aux, r[2]) / n * (1.0 if aux >= 0.0 else -1.0)
    return np.arctan2(sin_u, cos_u) % (2.0 * C.PI)


def transfer_velocity(theta: float, a: float, e: float,
                      i_tp: float, omega_node_tp: float, omega_tp: float) -> np.ndarray:
    """
    Скорость КА на орбите перелёта в гелиоцентрической системе OXYZ:
    составляющие (24), орбитальная система (25), переход матрицей A (7)-(9).
    """
    p = a * (1.0 - e ** 2)
    v_r = np.sqrt(C.MU_SUN / p) * e * np.sin(theta)
    v_tau = np.sqrt(C.MU_SUN / p) * (1.0 + e * np.cos(theta))
    v_orb = np.array([v_r * np.cos(theta) - v_tau * np.sin(theta),
                      v_r * np.sin(theta) + v_tau * np.cos(theta),
                      0.0])
    A = rotation_matrix(i_tp, omega_node_tp, omega_tp)
    return A @ v_orb


# внутренняя задача
def planetocentric_capture_dv(v_inf: float, mu_planet: float,
                              r_sphere: float, r_orbit: float) -> float:
    """
    Импульс скорости для ухода с круговой орбиты ИС (или торможения на неё)
    при гиперболическом избытке скорости v_inf, формула (29).
    """
    v_circ = np.sqrt(mu_planet / r_orbit)
    v_peri = np.sqrt(v_inf ** 2 - 2.0 * mu_planet / r_sphere + 2.0 * mu_planet / r_orbit)
    return v_peri - v_circ


# ----------------------------------------------------------------------------
# Полный расчёт траектории перелёта для варианта задания
# ----------------------------------------------------------------------------
def compute_transfer(variant: C.Variant) -> TransferResult:
    """
    Полный расчёт внешней и внутренней задач для заданного варианта:
    от даты старта и времени перелёта до суммарных затрат dV_сум.
    """
    planet = C.PLANETS.get(variant.planet)
    if planet is None:
        raise CalculationError(f"Неизвестная планета назначения: {variant.planet}")

    # Даты старта и прилёта
    D1 = julian_date_from_str(variant.date_start)
    D2 = D1 + variant.t12_days

    # Состояния Земли и планеты назначения в моменты D1 и D2 (п. 2.1)
    state1 = planet_state(C.EARTH, D1)
    state2 = planet_state(planet, D2)
    r1, r2 = state1.r_vec, state2.r_vec

    # Плоскость перелёта и угловая дальность (п. 2.2.1, формулы (10)-(12))
    i_tp, omega_node_tp = transfer_plane(r1, r2)
    phi = angular_range(r1, r2)
    chord_s = float(np.linalg.norm(r2 - r1))

    # Уравнение Ламберта (п. 2.2.2)
    t12_sec = variant.t12_days * C.SEC_PER_DAY
    t_gr, _ = boundary_transfer_time(state1.r, state2.r, chord_s, phi)
    a_tp, orbit_kind = solve_lambert_semimajor(state1.r, state2.r, chord_s, t12_sec, phi)
    e_tp, _eps, _delta, E1, E2 = eccentricity_and_anomalies(
        a_tp, state1.r, state2.r, chord_s, phi, orbit_kind)

    # Истинные аномалии точек 1 и 2 на орбите перелёта (21)
    theta1_tp = true_anomaly_from_E(E1, e_tp)
    theta2_tp = true_anomaly_from_E(E2, e_tp)

    # Аргумент перицентра омега_т.п., формулы (22)-(23)
    u1 = latitude_argument(r1, omega_node_tp)
    u2 = latitude_argument(r2, omega_node_tp)
    omega_tp = (u1 - theta1_tp) % (2.0 * C.PI)
    omega_check = (u2 - theta2_tp) % (2.0 * C.PI)
    d_omega = abs((omega_check - omega_tp + C.PI) % (2.0 * C.PI) - C.PI)
    if d_omega > 1e-6:
        raise CalculationError(
            f"Несогласованность аргумента перицентра по точкам 1 и 2: {d_omega * C.DEG:.3e} град")

    # Скорости КА в точках старта и прилёта (24)-(25) и переход к OXYZ
    v_sc1 = transfer_velocity(theta1_tp, a_tp, e_tp, i_tp, omega_node_tp, omega_tp)
    v_sc2 = transfer_velocity(theta2_tp, a_tp, e_tp, i_tp, omega_node_tp, omega_tp)

    # Импульсы скорости на границах сфер действия, формулы (26)-(28)
    dV1_vec = v_sc1 - state1.v_vec
    dV2_vec = v_sc2 - state2.v_vec
    dV1 = float(np.linalg.norm(dV1_vec))
    dV2 = float(np.linalg.norm(dV2_vec))

    # Внутренняя задача, формула (29)
    r_orb_earth = C.EARTH.radius + variant.h_isz_km
    r_orb_planet = planet.radius + variant.h_isp_km
    dV_isz = planetocentric_capture_dv(dV1, C.EARTH.mu, C.EARTH.r_sphere, r_orb_earth)
    dV_isp = planetocentric_capture_dv(dV2, planet.mu, planet.r_sphere, r_orb_planet)
    dV_total = dV_isz + dV_isp

    # Контроль: орбита перелёта должна проходить через точку r1
    p_tp = a_tp * (1.0 - e_tp ** 2)
    r1_check = p_tp / (1.0 + e_tp * np.cos(theta1_tp))
    r_orb = np.array([r1_check * np.cos(theta1_tp), r1_check * np.sin(theta1_tp), 0.0])
    A_tp = rotation_matrix(i_tp, omega_node_tp, omega_tp)
    residual = float(np.linalg.norm(A_tp @ r_orb - r1))

    return TransferResult(
        variant=variant, D1=D1, D2=D2, state1=state1, state2=state2,
        i_tp=i_tp, omega_node_tp=omega_node_tp, phi=phi, chord_s=chord_s,
        t_boundary=t_gr, orbit_kind=orbit_kind, a_tp=a_tp, e_tp=e_tp,
        omega_tp=omega_tp, theta1_tp=theta1_tp, theta2_tp=theta2_tp,
        v_sc1=v_sc1, v_sc2=v_sc2, dV1_vec=dV1_vec, dV2_vec=dV2_vec,
        dV1=dV1, dV2=dV2, dV_isz=dV_isz, dV_isp=dV_isp, dV_total=dV_total,
        check_residual_km=residual,
        )


# сканирование для энергетических карт
def _safe_compute(variant: C.Variant):
    """Расчёт варианта с перехватом ошибок (возвращает None при сбое)."""
    try:
        return compute_transfer(variant)
    except (CalculationError, ValueError):
        return None


def scan_departure_dates(variant: C.Variant,
                         half_range: float = C.DATE_SCAN_HALF_RANGE,
                         step: float = C.DATE_SCAN_STEP) -> List:
    """
    Сканирование по дате старта при фиксированной длительности перелёта.
    Возвращает список результатов (None — при несходимости в точке сетки).
    """
    D1_base = julian_date_from_str(variant.date_start)
    offsets = np.arange(-half_range, half_range + 0.5 * step, step)
    results = []
    for off in offsets:
        d, m, y = jd_to_date(D1_base + off)
        date_str = f"{d:02d}.{m:02d}.{y:04d}"
        results.append(_safe_compute(replace(variant, date_start=date_str)))
    return results


def scan_transfer_times(variant: C.Variant,
                        t_min: float = C.TIME_SCAN_MIN,
                        t_max: float = C.TIME_SCAN_MAX,
                        step: float = C.TIME_SCAN_STEP) -> List:
    """
    Сканирование по длительности перелёта при фиксированной дате старта.
    Возвращает список результатов (None — при несходимости в точке сетки).
    """
    times = np.arange(t_min, t_max + 0.5 * step, step)
    return [_safe_compute(replace(variant, t12_days=float(t))) for t in times]


def scan_grid(variant: C.Variant,
              n_date: int = C.CONTOUR_DATE_POINTS,
              n_time: int = C.CONTOUR_TIME_POINTS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Двумерное сканирование (дата старта x длительность перелёта) для построения
    изолиний суммарных характеристических затрат dV_сум.
    Возвращает (массив дат JD, массив t12, матрицу dV_сум с nan при сбоях).
    """
    D1_base = julian_date_from_str(variant.date_start)
    dates = np.linspace(D1_base - C.DATE_SCAN_HALF_RANGE,
                        D1_base + C.DATE_SCAN_HALF_RANGE, n_date)
    times = np.linspace(C.TIME_SCAN_MIN, C.TIME_SCAN_MAX, n_time)
    dv = np.full((n_time, n_date), np.nan)
    for j, D in enumerate(dates):
        d, m, y = jd_to_date(D)
        date_str = f"{d:02d}.{m:02d}.{y:04d}"
        for i, t in enumerate(times):
            res = _safe_compute(replace(variant, date_start=date_str, t12_days=float(t)))
            if res is not None:
                dv[i, j] = res.dV_total
    return dates, times, dv
