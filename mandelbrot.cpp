#include <SDL2/SDL.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <execution>
#include <functional>
#include <limits>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>

constexpr uint64_t max_iterations = 120;
constexpr double escape_radius = 4.0;

constexpr uint64_t window_width = 640;
constexpr uint64_t window_height = 480;
constexpr uint64_t pixel_count = window_width * window_height;

//no standard lib constexpr math funcs until c++ 26; sigh....
constexpr double ln_2 = 0.69314718056;

constexpr double aspect = static_cast<double>(window_width) / static_cast<double>(window_height);
constexpr double fractal_size = 4.0;
constexpr double scroll_increment = 0.05;
constexpr double zoom_limit = fractal_size - static_cast<int>(2*max_iterations-1) * scroll_increment;

constexpr std::size_t num_fps_samples = 10;
//defaults to number of physical cores available
std::size_t max_draw_threads = std::thread::hardware_concurrency();

constexpr std::size_t max_periodicity_checks = 20;

constexpr auto num_bench_iterations = 20;

constexpr auto max_zoom = 700;

struct RGB8 {
    uint8_t r, g, b;
};

inline double window_to_fractal_space_x(int w_x, double zoom) {
    return static_cast<double>(w_x) / static_cast<double>(window_width) * zoom * aspect - 0.5 * zoom * aspect;
}

inline double window_to_fractal_scale_x(int w_x, double zoom) {
    return static_cast<double>(w_x) / static_cast<double>(window_width) * zoom * aspect;
}

inline double window_to_fractal_space_y(int w_y, double zoom) {
    return static_cast<double>(w_y) / static_cast<double>(window_height) * zoom - 0.5 * zoom;
}

std::array<uint32_t, pixel_count> frame_buffer;

constexpr RGB8 hsv_to_rgb(double h, double s, double v) {
    double c = v * s;
    double hh = std::fmod(h / 60, 6);
    double x = c * (1 - std::fabs(std::fmod(hh, 2) - 1));
    double m = v - c;

    double r, g, b;
    if (0 <= hh && hh < 1) {
        r = c;
        g = x;
        b = 0;
    } else if (1 <= hh && hh < 2) {
        r = x;
        g = c;
        b = 0;
    } else if (2 <= hh && hh < 3) {
        r = 0;
        g = c;
        b = x;
    } else if (3 <= hh && hh < 4) {
        r = 0;
        g = x;
        b = c;
    } else if (4 <= hh && hh < 5) {
        r = x;
        g = 0;
        b = c;
    } else if (5 <= hh && hh < 6) {
        r = c;
        g = 0;
        b = x;
    } else {
        r = g = b = 0;
    }
    return RGB8{
        static_cast<uint8_t>((r + m) * 255),
        static_cast<uint8_t>((g + m) * 255),
        static_cast<uint8_t>((b + m) * 255)
    };
}

//generate color lookup table at compile time
constexpr auto color_table = [] {
    std::array<RGB8, 360> ct = {};
    for (std::size_t i = 0; i < 360; ++i) {
        ct[i] = hsv_to_rgb(static_cast<double>(i), 0.8, 0.8);
    }
    return ct;
 }();

template<typename T>
inline bool comp_epsilon(T a, T b) {
    return std::abs(a - b) < std::numeric_limits<T>::epsilon();
}

//#define CHECK_PERIODICITY //this really slows things down; not enough recursions..? ¯\_(ツ)_/¯
inline std::tuple<uint64_t, double>
mandelbrot(std::complex<double> c) {
    double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
    double cr = c.real(), ci = c.imag();
    uint64_t n = 0;
#ifdef CHECK_PERIODICITY
    double zr_old = 0.0, zi_old = 0.0;
    uint64_t period = 0;
#endif
    do {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        ++n;

#ifdef CHECK_PERIODICITY
        if (comp_epsilon(zr, zr_old) && comp_epsilon(zi, zi_old)) {
            n = max_iterations;
            break;
        }

        ++period;
        if (period > max_periodicity_checks) {
            period = 0.0;
            zr_old = zr;
            zi_old = zi;
        }
#endif
    } while (zr2 + zi2 <= escape_radius && n < max_iterations);
    return std::make_tuple(n, zr2 + zi2);
}

// void print_eight_pos(std::array<std::complex<double>, 8> pos) {
//     std::cout << "print pos\n";
//     for(auto c : pos) {
//         std::cout << pos << "\n";
//     }
// }

void mandelbrot_simd_blit(double t_x, double t_y, double t_zoom) {
    __m256d _zr, _zi, _zr2, _zi2, _two, _esc_rad, _x_scale, _a, _b;
    __m256d _cr, _ci, _x, _indices, _mask1, _x_stride;
    __m256i _iter, _max_iter, _one, _mask2, _c;

    double x_scale = aspect * t_zoom / static_cast<double>(window_width);
    //double y_scale = fractal_size / static_cast<double>(window_height);

    double x_start = window_to_fractal_space_x(0, t_zoom) + t_x;

    _x_scale = _mm256_set1_pd(x_scale);
    _x_stride = _mm256_set1_pd(x_scale*4.0);
    //_mask2 = _mm256_setzero_pd();
    _one = _mm256_set1_epi64x(1);
    _two = _mm256_set1_pd(2.0);
    _esc_rad = _mm256_set1_pd(escape_radius);
    _indices = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);
    _max_iter = _mm256_set1_epi64x(max_iterations);
    _indices = _mm256_mul_pd(_indices, _x_scale);

    //std::array<std::complex<double>, 8> pos;
    //auto idx = 0;

    for(auto j = 0; j < window_height; ++j) {
        double y = window_to_fractal_space_y(j, t_zoom) + t_y;
        _ci = _mm256_set1_pd(y);
        _a = _mm256_set1_pd(x_start);
        _x = _mm256_add_pd(_a, _indices);
        for(auto i = 0; i < window_width; i += 4) {
            // if(j == 0 && (i == 0 || i == 4)) {
            //     pos[4*idx] = _x[0];
            //     pos[4*idx+1] = _x[1];
            //     pos[4*idx+2] = _x[2];
            //     pos[4*idx+3] = _x[3];
            // }
            _cr = _x;
            _zr = _zi = _mm256_setzero_pd();
            _iter = _mm256_setzero_si256();

        loop:
            _zr2 = _mm256_mul_pd(_zr, _zr);
            _zi2 = _mm256_mul_pd(_zi, _zi);
            _a = _mm256_sub_pd(_zr2, _zi2);
            _a = _mm256_add_pd(_a, _cr);

            _b = _mm256_mul_pd(_zr, _zi);
            _b = _mm256_fmadd_pd(_two, _b, _ci);
            _zr = _a;
            _zi = _b;

            _a = _mm256_add_pd(_zr2, _zi2);

            _mask1 = _mm256_cmp_pd(_a, _esc_rad, _CMP_LT_OQ);
            _mask2 = _mm256_cmpgt_epi64(_max_iter, _iter);
            _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));

            _c = _mm256_and_si256(_one, _mask2);

            _iter = _mm256_add_epi64(_iter, _c);

            if(_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0) goto loop;
            //loop done
            _a = _mm256_sqrt_pd(_a);

            for(auto n = 0; n < 4; ++n) {
                RGB8 color = RGB8{0, 0, 0};
                if(static_cast<uint64_t>(_iter[n]) < max_iterations) {
                    double i_adj = static_cast<double>(_iter[n]) + 1 - std::log(std::log(_a[n]));
                    double value = std::min(i_adj / static_cast<double>(max_iterations), 1.0);
                    std::size_t index = static_cast<uint8_t>(value * 360.0);
                    color = color_table[359 - (index - 1)];
                }
                uint32_t pixel = (uint32_t(color.r) << 24) | (uint32_t(color.g) << 16) | (uint32_t(color.b) << 8);
                frame_buffer[j * window_width + i + 3 - n] = pixel;
            }
            _x = _mm256_add_pd(_x, _x_stride);
        }
        //y_offset += window_width;
    }
}

void mandelbrot_simd_omp_blit(double t_x, double t_y, double t_zoom) {
    __m256d _zr, _zi, _zr2, _zi2, _two, _esc_rad, _x_scale, _a, _b;
    __m256d _cr, _ci, _x, _indices, _mask1, _x_stride;
    __m256i _iter, _max_iter, _one, _mask2, _c;

    double x_scale = aspect * t_zoom / static_cast<double>(window_width);
    //double y_scale = fractal_size / static_cast<double>(window_height);

    double x_start = window_to_fractal_space_x(0, t_zoom) + t_x;

    _x_scale = _mm256_set1_pd(x_scale);
    _x_stride = _mm256_set1_pd(x_scale*4.0);
    //_mask2 = _mm256_setzero_pd();
    _one = _mm256_set1_epi64x(1);
    _two = _mm256_set1_pd(2.0);
    _esc_rad = _mm256_set1_pd(escape_radius);
    _indices = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);
    _max_iter = _mm256_set1_epi64x(max_iterations);
    _indices = _mm256_mul_pd(_indices, _x_scale);

    //std::array<std::complex<double>, 8> pos;
    //auto idx = 0;
#pragma omp parallel for schedule(dynamic) num_threads(max_draw_threads)
    for(auto j = 0; j < window_height; ++j) {
        double y = window_to_fractal_space_y(j, t_zoom) + t_y;
        _ci = _mm256_set1_pd(y);
        _a = _mm256_set1_pd(x_start);
        _x = _mm256_add_pd(_a, _indices);
        for(auto i = 0; i < window_width; i += 4) {
            // if(j == 0 && (i == 0 || i == 4)) {
            //     pos[4*idx] = _x[0];
            //     pos[4*idx+1] = _x[1];
            //     pos[4*idx+2] = _x[2];
            //     pos[4*idx+3] = _x[3];
            // }
            _cr = _x;
            _zr = _zi = _mm256_setzero_pd();
            _iter = _mm256_setzero_si256();

        loop:
            _zr2 = _mm256_mul_pd(_zr, _zr);
            _zi2 = _mm256_mul_pd(_zi, _zi);
            _a = _mm256_sub_pd(_zr2, _zi2);
            _a = _mm256_add_pd(_a, _cr);

            _b = _mm256_mul_pd(_zr, _zi);
            _b = _mm256_fmadd_pd(_two, _b, _ci);
            _zr = _a;
            _zi = _b;

            _a = _mm256_add_pd(_zr2, _zi2);

            _mask1 = _mm256_cmp_pd(_a, _esc_rad, _CMP_LT_OQ);
            _mask2 = _mm256_cmpgt_epi64(_max_iter, _iter);
            _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));

            _c = _mm256_and_si256(_one, _mask2);

            _iter = _mm256_add_epi64(_iter, _c);

            if(_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0) goto loop;
            //loop done
            _a = _mm256_sqrt_pd(_a);

            //std::array<double, 4> c_abs; _mm256_store_pd(c_abs.data(), _a);
            //std::array<long long, 4> iterations; _mm256_maskstore_epi64(iterations.data(), _iter, );

            for(auto n = 0; n < 4; ++n) {
                RGB8 color = RGB8{0, 0, 0};
                if(_iter[n] < max_iterations) {
                    double i_adj = static_cast<double>(_iter[3 - n]) + 1 - std::log(std::log(_a[3 - n]));
                    double value = std::min(i_adj / static_cast<double>(max_iterations), 1.0);
                    std::size_t index = static_cast<uint8_t>(value * 360.0);
                    color = color_table[359 - (index - 1)];
                }
                uint32_t pixel = (uint32_t(color.r) << 24) | (uint32_t(color.g) << 16) | (uint32_t(color.b) << 8);
                frame_buffer[j * window_width + i + n] = pixel;
            }
            _x = _mm256_add_pd(_x, _x_stride);
        }
        //y_offset += window_width;
    }
}

void mandelbrot_blit(double t_x, double t_y, double t_zoom) {
    for(auto j = 0; j < window_height; ++j) {
        double y = window_to_fractal_space_y(j, t_zoom) + t_y;
        for(auto i = 0; i < window_width; ++i) {
            double x = window_to_fractal_space_x(i, t_zoom) + t_x;
            auto [iterations, norm] = mandelbrot(std::complex<double>(x, y));
            RGB8 color = RGB8{0, 0, 0};
            if(iterations < max_iterations) {
                double i_adj = static_cast<double>(iterations) + 1 - std::log(std::log(std::sqrt(norm))) / ln_2;
                double value = std::min(i_adj / static_cast<double>(max_iterations), 1.0);
                std::size_t index = static_cast<uint8_t>(value * 360.0);
                color = color_table[359 - (index - 1)];
            }
            uint32_t pixel = (uint32_t(color.r) << 24) | (uint32_t(color.g) << 16) | (uint32_t(color.b) << 8);
            frame_buffer[j*window_width+i] = pixel;
        }
    }
}

void mandelbrot_blit_omp(double t_x, double t_y, double t_zoom) {
#pragma omp parallel for schedule(dynamic) num_threads(max_draw_threads)
    for(auto j = 0; j < window_height; ++j) {
        double y = window_to_fractal_space_y(j, t_zoom) + t_y;
        for(auto i = 0; i < window_width; ++i) {
            double x = window_to_fractal_space_x(i, t_zoom) + t_x;
            auto [iterations, norm] = mandelbrot(std::complex<double>(x, y));
            RGB8 color = RGB8{0, 0, 0};
            if(iterations < max_iterations) {
                double i_adj = static_cast<double>(iterations) + 1 - std::log(std::log(std::sqrt(norm))) / ln_2;
                double value = std::min(i_adj / static_cast<double>(max_iterations), 1.0);
                std::size_t index = static_cast<uint8_t>(value * 360.0);
                color = color_table[359 - (index - 1)];
            }
            uint32_t pixel = (uint32_t(color.r) << 24) | (uint32_t(color.g) << 16) | (uint32_t(color.b) << 8);
            frame_buffer[j*window_width+i] = pixel;
        }
    }
}

std::mutex color_table_mutex;
RGB8 color_table_thread(std::size_t idx) {
    std::lock_guard<std::mutex> lock(color_table_mutex);
    RGB8 color = color_table[idx];
    return color;
}

std::mutex printf_mutex;
void printf_ts(const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(printf_mutex);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
}

void mandelbrot_thread(size_t start_y, size_t end_y, double t_x, double t_y, double t_zoom) {
    for(auto j = start_y; j < end_y; ++j) {
        double y = window_to_fractal_space_y(j, t_zoom) + t_y;
        for(auto i = 0; i < window_width; ++i) {
            double x = window_to_fractal_space_x(i, t_zoom) + t_x;
            auto [iterations, norm] = mandelbrot(std::complex<double>(x, y));
            RGB8 color = RGB8{0, 0, 0};
            if(iterations < max_iterations) {
                double i_adj = static_cast<double>(iterations) + 1 - std::log(std::log(std::sqrt(norm))) / ln_2;
                double value = std::min(i_adj / static_cast<double>(max_iterations), 1.0);
                std::size_t index = static_cast<uint8_t>(value * 360.0);
                color = color_table_thread(359 - (index - 1));
            }
            uint32_t pixel = (uint32_t(color.r) << 24) | (uint32_t(color.g) << 16) | (uint32_t(color.b) << 8);
            frame_buffer[j*window_width+i] = pixel;
        }
    }
}

void mandelbrot_blit_threaded_bad(double t_x,
                              double t_y,
                              double t_zoom,
                              std::size_t num_draw_threads) {
    assert(num_draw_threads <= max_draw_threads && num_draw_threads > 0);
    std::size_t pos_offset = window_height / num_draw_threads;
    std::vector<std::jthread> draw_threads;
    for (auto t = 0; t < num_draw_threads; ++t) {
        draw_threads.emplace_back(std::jthread(mandelbrot_thread,
                                               t*pos_offset,
                                               (t+1)*pos_offset,
                                               t_x, t_y, t_zoom));
    }
}

void mandelbrot_flip(SDL_Renderer* renderer, SDL_Texture* frame_texture) {
    SDL_UpdateTexture(frame_texture, NULL, frame_buffer.data(), window_width * 4);
    SDL_RenderCopy(renderer, frame_texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

template<class Fn, class... Args>
void benchmark(std::string name, Fn func, Args... args) {
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < num_bench_iterations; ++i) {
        func(std::forward<Args>(args)...);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg = elapsed.count() / static_cast<double>(num_bench_iterations);
    std::cout << name << " finished execution after " << elapsed.count() << " ms.\n";
    std::cout << "average execution time: " << avg << " ms\n";
}

int main(int argc, char** argv) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Mandelbrot",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          window_width,
                                          window_height,
                                          SDL_WINDOW_SHOWN);

    SDL_Renderer* renderer = SDL_CreateRenderer(window, 0, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

     SDL_Event e;
     bool run = true;

     SDL_Texture* frame_texture = SDL_CreateTexture(renderer,
                                                    SDL_PIXELFORMAT_RGBA8888,
                                                    SDL_TEXTUREACCESS_STREAMING,
                                                    window_width,
                                                    window_height);

     //benchmark("single threaded", mandelbrot_blit, 0.0, 0.0, 4.0);
     //benchmark("simd", mandelbrot_simd_blit, 0.0, 0.0, 4.0);
     //benchmark("simd omp", mandelbrot_simd_omp_blit, 0.0, 0.0, 4.0);
     //benchmark("multi threaded", mandelbrot_blit_threaded_bad, 0.0, 0.0, 4.0, max_draw_threads);
     //benchmark("openmp", mandelbrot_blit_omp, 0.0, 0.0, 4.0);

     std::cout << "# of threads: " << max_draw_threads << "\n";

     std::chrono::time_point<std::chrono::steady_clock> start, end;

     double t_x = 0.0, t_y = 0.0, t_zoom = 4.0;
     int scroll = 0;

     std::array<double, num_fps_samples> frame_times = {0.0};
     std::size_t time_idx = 0;
     double min_fps_avg = std::numeric_limits<double>::max(), max_fps_avg = 0;


     while (run) {
         start = std::chrono::steady_clock::now();
         while (SDL_PollEvent(&e)) {
             switch (e.type) {
             case SDL_QUIT:
                 run = false;
                 break;
             case SDL_KEYDOWN:
                 if (e.key.keysym.sym == SDLK_ESCAPE) {
                     run = false;
                 }
                 break;
             case SDL_MOUSEWHEEL:
                 scroll += e.wheel.y;
                 //if you zoom any further, then camera movement no longer works due to lack of precision
                 scroll = std::clamp(scroll, 0, max_zoom);
                 break;
             }
         }
         int t_mdx, t_mdy;
         uint32_t m_b = SDL_GetRelativeMouseState(&t_mdx, &t_mdy);
         //t_zoom = fractal_size * (1.0 / std::exp(static_cast<double>(scroll) * scroll_increment));
         t_zoom = fractal_size * (std::exp(static_cast<double>(-scroll) * scroll_increment));
         t_zoom = std::clamp(t_zoom, std::numeric_limits<double>::min(), fractal_size);

         if (m_b & SDL_BUTTON(1) || m_b & SDL_BUTTON(2) || m_b & SDL_BUTTON(3)) {
             double m_dx_p = window_to_fractal_space_x(t_mdx, t_zoom) + 0.5*t_zoom*aspect;
             double m_dy_p = window_to_fractal_space_y(t_mdy, t_zoom) + 0.5*t_zoom;
             t_x -= m_dx_p; t_y -= m_dy_p;
         }

         t_x = std::clamp(t_x, -fractal_size*0.5, fractal_size*0.5);
         t_y = std::clamp(t_y, -fractal_size*0.5, fractal_size*0.5);

         //#define USE_THREADED_HYBRID
         //#define USE_THREADED_ONLY
         //#define USE_SINGLE_THREADED
         #define USE_PARALLEL_OMP
#ifdef USE_THREADED_HYBRID
         if (scroll < max_iterations/2) {
             mandelbrot_blit(t_x, t_y, t_zoom);
             end = std::chrono::steady_clock::now();
         } else {
             mandelbrot_blit_threaded_bad(t_x, t_y, t_zoom, max_draw_threads);
         }
#endif
#ifdef USE_THREADED_ONLY
         mandelbrot_blit_threaded_bad(t_x, t_y, t_zoom, max_draw_threads);
#endif
#ifdef USE_SINGLE_THREADED
         //mandelbrot_blit(t_x, t_y, t_zoom);
         mandelbrot_simd_blit(t_x, t_y, t_zoom);
#endif
#ifdef USE_PARALLEL_OMP
         mandelbrot_blit_omp(t_x, t_y, t_zoom);
#endif
         mandelbrot_flip(renderer, frame_texture);

         end = std::chrono::steady_clock::now();
         std::chrono::duration<double, std::milli> elapsed = end - start;

         frame_times[time_idx++] = elapsed.count();
         time_idx = frame_times.size() % time_idx;

         double frame_time_avg = std::accumulate(frame_times.begin(), frame_times.end(), 0.0);
         if (frame_time_avg > max_fps_avg) max_fps_avg = frame_time_avg;
         if (frame_time_avg < min_fps_avg) min_fps_avg = frame_time_avg;
         std::cout << "\r" << "average frame time: " << frame_time_avg << " ms; zoom level: " << scroll << " t_zoom: " << t_zoom << std::flush;
     }
     printf("\nmax frame time average: %f; min frame time average: %f\n", max_fps_avg, min_fps_avg);

     SDL_DestroyTexture(frame_texture);
     SDL_DestroyRenderer(renderer);
     SDL_DestroyWindow(window);
     SDL_Quit();
     return 0;
}
