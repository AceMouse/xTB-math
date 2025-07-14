#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <filesystem>
#include <stdlib.h>
using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " <binary float64 file> <number of fullerenes>\n";
    return EXIT_FAILURE;
  }

  ifstream file(argv[1], ios::binary);
  if (!file) {
      cerr << "Error: Cannot open file\n";
      return 1;
  }

  // Determine file size
  file.seekg(0, ios::end);
  streamsize size = file.tellg();
  file.seekg(0, ios::beg);

  vector<double> molecule(size / sizeof(double));

  // Read binary data into vector
  if (!file.read(reinterpret_cast<char*>(molecule.data()), size)) {
      cerr << "Error: Failed to read file\n";
      return 1;
  }

  file.close();

  filesystem::create_directory("output");

  int num_fullerenes = atoi(argv[2]);
  int num_atoms = molecule.size() / num_fullerenes / 3;
  for (int a = 0; a < num_fullerenes; a++) {
    stringstream ss;
    ss << "output/output_" << a << ".xyz";
    ofstream file_out(ss.str());
    if (!file_out) {
        cerr << "Failed to open file for writing.\n";
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
