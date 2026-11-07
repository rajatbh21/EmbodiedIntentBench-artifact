// Export installed Gazebo's parsed triangles and normal/primitive metadata.
#include <gz/common/MeshManager.hh>
#include <gz/common/Mesh.hh>
#include <gz/common/SubMesh.hh>
#include <fstream>
#include <iomanip>
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  auto mesh = gz::common::MeshManager::Instance()->Load(argv[1]);
  if (!mesh) return 3;
  std::ofstream coordinates(argv[2]), metadata(argv[3]);
  coordinates << std::setprecision(17);
  for (unsigned i = 0; i < mesh->SubMeshCount(); ++i) {
    auto submesh = mesh->SubMeshByIndex(i).lock();
    metadata << submesh->Name() << "," << submesh->VertexCount() << ","
             << submesh->NormalCount() << "," << submesh->IndexCount() << ","
             << (submesh->SubMeshPrimitiveType() == gz::common::SubMesh::TRIANGLES) << "\n";
    if (submesh->IndexCount() % 3) return 4;
    for (unsigned j = 0; j < submesh->IndexCount(); j += 3) {
      for (unsigned k = 0; k < 3; ++k) {
        auto vertex = submesh->Vertex(submesh->Index(j + k));
        if (k) coordinates << ",";
        coordinates << vertex.X() << "," << vertex.Y() << "," << vertex.Z();
      }
      coordinates << "\n";
    }
  }
}
