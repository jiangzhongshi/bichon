#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

void feature_and_curve(std::string filename, std::string fgname,
                       std::string ser_file, nlohmann::json config);

auto dict_to_option(nlohmann::json &config, CLI::App &program) -> void {
  for (auto &[cmd, subopt] : config.items()) {
    for (auto &[key, val] : subopt.items()) {
      if (val.is_boolean()) {
        program.add_flag(
            fmt::format("--{}-{}, !--{}-no-{}", cmd, key, cmd, key),
            val.get_ref<bool &>());
      } else if (val.is_number_integer()) {
        program.add_option(fmt::format("--{}-{}", cmd, key),
                           val.get_ref<nlohmann::json::number_integer_t &>(),
                           "int");
      } else if (val.is_number_float()) {
        program.add_option(fmt::format("--{}-{}", cmd, key),
                           val.get_ref<nlohmann::json::number_float_t &>(),
                           "double");
      } else {
        throw std::runtime_error("Option Type not implemented.");
      }
    }
  }
}

int main(int argc, char **argv) {
  CLI::App program{"Bijective and Coarse High-Order Tetrahedral Meshes."};

  std::string filename, output_dir = "./", input_file, feature_graph_file,
                        log_dir = "";
  program.add_option("-i,--input", input_file, "input mesh name")
      ->required()
      ->check(CLI::ExistingFile)
      ->each([&filename](const std::string &s) {
        filename = std::filesystem::path(s).filename().string();
      });
  program.add_option("-g,--graph", feature_graph_file, "feature graph .fgraph");
  program.add_option("-o,--output", output_dir, "output dir")
      ->default_str("./");
  program.add_option<std::string>("-l,--logdir", log_dir, "log dir");
  program.add_option_function<int>(
      "--loglevel",
      [](const int &l) {
        spdlog::set_level(static_cast<spdlog::level::level_enum>(l));
      },
      "log level");

  std::string suffix = "";
  program.add_option("--suffix", suffix, "suffix identifier");

  auto config = nlohmann::json();
  config["curve"] = {{"order", 3},
                     {"distance_threshold", 1e-2},
                     {"normal_threshold", -1.0},
                     {"recursive_check", true}};
  config["shell"] = {{"initial_thickness", 1e-2},
                     {"target_edge_length", 1e-1},
                     {"distortion_bound", 0.01},
                     {"target_thickness", 5e-2}};
  config["feature"] = {
      {"enable_polyshell", false},
      {"initial_split_edge", 2e-1},
      {"dihedral_threshold", 0.5}  // 120 degree.
  };
  config["control"] = {
      {"enable_curve", true},
      {"reset_cp", false},  // this is a experiment switch: reset linear cp so
                            // that we can load a un-curved intermediate model.
      {"serialize_level", 2},
      {"freeze_feature", false},
      {"only_initial", false},
      {"skip_collapse", false},
      {"skip_split", true},
      {"skip_volume", false},
  };
  config["tetfill"] = {{"tetwild", true}};
  config["cutet"] = {
      {"debug", false},
      {"passes", 6},
      {"smooth_iter", 4},
      {"energy_threshold", 100},
  };
  dict_to_option(config, program);

  program.callback([&]() {
    filename = std::filesystem::path(input_file).filename().string();
    if (log_dir != "") {
      auto file_logger = spdlog::basic_logger_mt(
          "cumin", log_dir + "/" + filename + suffix + ".log");
      spdlog::set_default_logger(file_logger);
    }
    spdlog::flush_on(spdlog::level::info);
    spdlog::info("{}", config.dump());
    feature_and_curve(input_file, feature_graph_file,
                      output_dir + "/" + filename + suffix + ".h5", config);
  });

  CLI11_PARSE(program, argc, argv);
}
