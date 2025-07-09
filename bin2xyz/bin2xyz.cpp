#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <filesystem>

int main() {
  std::ifstream file("C200_10000_fullerenes.float64", std::ios::binary);
  if (!file) {
      std::cerr << "Error: Cannot open file\n";
      return 1;
  }

  // Determine file size
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<double> molecule(size / sizeof(double));

  // Read binary data into vector
  if (!file.read(reinterpret_cast<char*>(molecule.data()), size)) {
      std::cerr << "Error: Failed to read file\n";
      return 1;
  }

  file.close();

  std::filesystem::create_directory("output");

  int num_atoms = molecule.size() / 10000 / 3;
  for (int a = 0; a < 10000; a++) {
    std::stringstream ss;
    ss << "output/output_" << a << ".xyz";
    std::ofstream file_out(ss.str());
    if (!file_out) {
        std::cerr << "Failed to open file for writing.\n";
        return 1;
    }
    file_out << num_atoms << "\n\n";
    for (int i = 0; i < num_atoms; ++i) {
        float x = molecule[num_atoms * a + 3 * i];
        float y = molecule[num_atoms * a + 3 * i + 1];
        float z = molecule[num_atoms * a + 3 * i + 2];
        file_out << "C " << x << " " << y << " " << z << "\n";
    }
    file_out.close();
  }
}
