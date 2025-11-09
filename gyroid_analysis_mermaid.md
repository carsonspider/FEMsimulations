# Gyroid Analysis Flowchart
```mermaid
flowchart TD
    %% Entry point
    A[Start Script] --> B{External STL path passed?}
    %% External STL branch
    B -- Yes --> C[Create SimpleGyroidAnalysis()
size=20, resolution=60]
    C --> D[load_external_stl(stl_path, resolution=80)]
    D --> D1[voxelize mesh via _voxelize_mesh()
ray casting per grid point]
    D1 --> E[simulate_compressive_loading(1 MPa)]
    E --> E1[calculate_mechanical_properties()
Gibson-Ashby scaling]
    E1 --> F[analyze_stress_distribution()
neighbor counting]
    F --> G[visualize_results()
2D slices, 3D scatter, histogram]
    G --> H[report_analysis()
print geometry & mechanics]
    H --> I[Print completion with STL path]
    I --> Z[End]
    %% Generated gyroid branch
    B -- No --> J[Print default parameters]
    J --> K[Create SimpleGyroidAnalysis(
size=20, resolution=100,
unit_cell_size=3, thickness=0.7,
smoothness=0.8)]
    K --> L[generate_gyroid()
Gaussian-smoothed level set]
    L --> L1[Binary opening & closing
cleanup volume]
    L1 --> M[create_mesh()
scipy marching_cubes]
    M --> N[save_stl()
write gyroid_<timestamp>.stl]
    N --> O[simulate_compressive_loading(20 MPa)]
    O --> O1[calculate_mechanical_properties()
relative density → stiffness]
    O1 --> P[analyze_stress_distribution()]
    P --> Q[visualize_results()
fig with 6 subplots]
    Q --> R[report_analysis()
print summary]
    R --> S[Print completion with STL path]
    S --> Z
```
