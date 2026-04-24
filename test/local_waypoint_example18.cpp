#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include <ruckig/ruckig.hpp>

using namespace ruckig;

using Vec3 = std::array<double, 3>;

static InputParameter<3> make_input() {
    InputParameter<3> inp;
    inp.current_position = {0.2, 0.0, -0.3};
    inp.current_velocity = {0.0, 0.0, 0.0};
    inp.current_acceleration = {0.0, 0.0, 0.0};

    inp.intermediate_positions = {
        Vec3{1.4, -1.6, 1.0},
        Vec3{-0.6, -0.5, 0.4},
        Vec3{-0.4, -0.35, 0.0},
        Vec3{0.8, 1.8, -0.1},
    };

    inp.target_position = {0.5, 1.0, 0.0};
    inp.target_velocity = {0.0, 0.0, 0.0};
    inp.target_acceleration = {0.0, 0.0, 0.0};

    inp.max_velocity = {1.0, 2.0, 1.0};
    inp.max_acceleration = {3.0, 2.0, 2.0};
    inp.max_jerk = {6.0, 10.0, 20.0};

    return inp;
}

static std::vector<Vec3> build_reference_polyline(const InputParameter<3>& inp) {
    std::vector<Vec3> pts;
    pts.push_back(inp.current_position);
    for (const auto& wp: inp.intermediate_positions) {
        pts.push_back(wp);
    }
    pts.push_back(inp.target_position);
    return pts;
}

static double norm(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

static Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

static Vec3 mul(const Vec3& a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

static double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static double point_to_segment_dist(const Vec3& p, const Vec3& a, const Vec3& b) {
    const Vec3 ab = sub(b, a);
    const double len_sq = dot(ab, ab);
    if (len_sq < 1e-18) {
        return norm(sub(p, a));
    }
    const double t = std::clamp(dot(sub(p, a), ab) / len_sq, 0.0, 1.0);
    const Vec3 closest = add(a, mul(ab, t));
    return norm(sub(p, closest));
}

static std::vector<Vec3> resample_positions_uniform(const std::vector<Vec3>& pos, size_t n_samples) {
    if (pos.size() < 2) {
        return pos;
    }

    std::vector<double> arc(pos.size(), 0.0);
    for (size_t i = 1; i < pos.size(); ++i) {
        arc[i] = arc[i - 1] + norm(sub(pos[i], pos[i - 1]));
    }

    const double total_arc = arc.back();
    if (total_arc < 1e-12) {
        return pos;
    }

    std::vector<Vec3> out(n_samples);
    size_t src_idx = 0;
    for (size_t k = 0; k < n_samples; ++k) {
        const double s = total_arc * static_cast<double>(k) / static_cast<double>(n_samples - 1);
        while (src_idx + 1 < arc.size() && arc[src_idx + 1] < s) {
            ++src_idx;
        }

        if (src_idx + 1 >= arc.size()) {
            out[k] = pos.back();
            continue;
        }

        const double s0 = arc[src_idx];
        const double s1 = arc[src_idx + 1];
        const double alpha = (s1 > s0) ? (s - s0) / (s1 - s0) : 0.0;
        out[k] = add(pos[src_idx], mul(sub(pos[src_idx + 1], pos[src_idx]), alpha));
    }

    return out;
}

static std::vector<double> path_deviations(const std::vector<Vec3>& positions,
                                           const std::vector<Vec3>& ref_poly) {
    std::vector<double> dists(positions.size(), std::numeric_limits<double>::infinity());
    for (size_t seg = 0; seg + 1 < ref_poly.size(); ++seg) {
        for (size_t i = 0; i < positions.size(); ++i) {
            dists[i] = std::min(dists[i], point_to_segment_dist(positions[i], ref_poly[seg], ref_poly[seg + 1]));
        }
    }
    return dists;
}

static void report_stats(const std::vector<double>& dev) {
    std::vector<double> sorted = dev;
    std::sort(sorted.begin(), sorted.end());
    const double max_v = sorted.back();
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    double sq = 0.0;
    for (double v: sorted) sq += v * v;
    const double rms = std::sqrt(sq / static_cast<double>(sorted.size()));
    const double p95 = sorted[static_cast<size_t>(0.95 * static_cast<double>(sorted.size() - 1))];
    std::printf("local max=%.6f mean=%.6f rms=%.6f p95=%.6f\n", max_v, mean, rms, p95);
}

int main() {
    Ruckig<3> otg(0.01, 10);
    otg.set_waypoints_backend(WaypointsBackend::Local);

    auto inp = make_input();
    OutputParameter<3> out(10);

    std::vector<Vec3> positions;
    positions.push_back(inp.current_position);

    Result res = Result::Working;
    while (res == Result::Working) {
        res = otg.update(inp, out);
        positions.push_back(out.new_position);
        out.pass_to_input(inp);
    }

    if (res < 0) {
        std::printf("local backend failed with code %d\n", static_cast<int>(res));
        return 1;
    }

    const auto ref_poly = build_reference_polyline(make_input());
    const auto resampled = resample_positions_uniform(positions, 2000);
    const auto dev = path_deviations(resampled, ref_poly);
    report_stats(dev);
    return 0;
}
