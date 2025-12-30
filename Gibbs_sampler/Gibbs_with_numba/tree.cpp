// tree.cpp
#include "pybind11_tree.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// Constructor / destructor
Tree::Tree(int N_, int M_, int dimx_) : N(N_), M(M_), dimx(dimx_) {
    if (M < 3 * N) M = 3 * N;
    nsteps = 0;
    a_star = VectorXi::Zero(M);
    o_star = VectorXi::Zero(M);
    x_star = MatrixXd::Zero(dimx, M);
    l_star = VectorXi::Zero(N);
    reset();
}

Tree::~Tree() {}

void Tree::reset() {
    nsteps = 0;
    a_star.setZero();
    o_star.setZero();

    // parent links: use negative sentinel so get_path stops (consistent with init())
    //a_star.setConstant(-2);   //init uses -2; be consistent
    // no offspring initially
    //o_star.setZero();

    // reset leaf mapping to trivial 0..N-1: recommended for a fresh tree
    for (int i = 0; i < N; ++i) l_star(i) = i;

    // zero stored states (optional but safe)
    x_star.setZero();
}

// Initialize with an Eigen matrix of shape (dimx, N)
void Tree::init(const MatrixXd& x_0) {
    // assume x_0 is (dimx x N)
    for (int i = 0; i < N; ++i) {
        a_star(i) = -2;
        l_star(i) = i;
        x_star.col(i) = x_0.col(i);
    }
}

void Tree::insert(const MatrixXd& x, const VectorXi& a) {
    nsteps++;
    // gather parents: b[i] = l_star[a(i)]
    VectorXi b(N);
    for (int i = 0; i < N; ++i) b(i) = l_star(a(i));

    // find empty slots (where o_star == 0)
    int slot = 0;
    int i = 0;
    while (slot < N && i < M) {
        if (o_star(i) == 0) {
            l_star(slot++) = i;
        }
        ++i;
    }

    if (slot < N) {
        int old_M = M;
        double_size();
        for (i = slot; i < N; ++i) l_star(i) = old_M + i - slot;
    }

    // scatter: place parents and particles into a_star / x_star at l_star positions
    for (int i = 0; i < N; ++i) {
        int idx = l_star(i);
        a_star(idx) = b(i);
        x_star.col(idx) = x.col(i);
    }
}

void Tree::prune(const VectorXi& o) {
    // scatter o into o_star at the leaf indices
    for (int i = 0; i < N; ++i) o_star(l_star(i)) = o(i);

    // walk backward from leaves and decrement offspring counts; prune when zero
    for (int i = 0; i < N; ++i) {
        int j = l_star(i);
        while (j >= 0 && o_star(j) == 0) {
            j = a_star(j);
            if (j >= 0) o_star(j) = o_star(j) - 1;
        }
    }
}

void Tree::update(const MatrixXd& x, const VectorXi& a) {
    // convert ancestor vector to offspring counts
    VectorXi o = VectorXi::Zero(N);
    for (int i = 0; i < N; ++i) {
        o(a(i)) += 1;
    }
    // prune then insert
    prune(o);
    insert(x, a);
}

void Tree::double_size() {
    int old_M = M;
    M = 2 * old_M;
    // resize vectors/matrices preserving existing data
    a_star.conservativeResize(M);
    o_star.conservativeResize(M);

    MatrixXd new_x = MatrixXd::Zero(dimx, M);
    new_x.block(0, 0, dimx, old_M) = x_star;
    x_star.swap(new_x);
}

MatrixXd Tree::get_path(int n) {
    // returns (dimx, nsteps + 1)
    MatrixXd path = MatrixXd::Zero(dimx, nsteps + 1);
    int j = l_star(n);
    path.col(nsteps) = x_star.col(j);
    int step = nsteps - 1;
    while (j >= 0 && step >= 0) {
        j = a_star(j);
        if (j < 0) break;
        path.col(step) = x_star.col(j);
        --step;
    }
    return path;
}

MatrixXd Tree::retrieve_xgeneration(int lag) {
    MatrixXd Xgen = MatrixXd::Zero(dimx, N);
    for (int i_particle = 0; i_particle < N; ++i_particle) {
        int j = l_star(i_particle);
        for (int i_step = 0; i_step < lag; ++i_step) j = a_star(j);
        Xgen.col(i_particle) = x_star.col(j);
    }
    return Xgen;
}

void Tree::reconstruct(int N_, int M_, int d, int n,
                       const VectorXi& a, const VectorXi& o,
                       const MatrixXd& x, const VectorXi& l) {
    this->N = N_;
    this->M = M_;
    this->dimx = d;
    this->nsteps = n;
    this->a_star = a;
    this->o_star = o;
    this->x_star = x;
    this->l_star = l;
}


void Tree::bulk_load(const MatrixXd& full_traj, const MatrixXd& full_ancestors) {
    // 1. Reset
    this->reset();
    
    // Validate dimensions (Optional debug check)
    int T = full_ancestors.cols(); 
    // full_traj should have N * T columns
    // We assume N matches this->N
     
    // 2. Init with Time t=0
    // We take the first block: Rows 0..dimx, Cols 0..N
    // .block(start_row, start_col, num_rows, num_cols)
    MatrixXd x_0 = full_traj.block(0, 0, dimx, N);
    this->init(x_0);
    
    // 3. Loop from t=1 to T-2 (Insert)
    for (int t = 1; t < T - 1; ++t) {
        // Ancestors: Column t of the ancestor matrix
        VectorXi a_t = full_ancestors.col(t).cast<int>();
        
        // Trajectory: The t-th block of N columns
        // Start column is t * N
        MatrixXd x_t = full_traj.block(0, t * N, dimx, N);
        
        this->insert(x_t, a_t);
    }
    
    // 4. Final Update (t = T-1)
    if (T > 1) {
        VectorXi a_last = full_ancestors.col(T - 1).cast<int>();
        MatrixXd x_last = full_traj.block(0, (T - 1) * N, dimx, N);
        this->update(x_last, a_last);
    }
}


// Expose the class to Python (pybind11 will convert Eigen <-> NumPy)
PYBIND11_MODULE(tree, m) {
    py::class_<Tree>(m, "Tree")
        .def(py::init<int,int,int>())
        .def_readwrite("N", &Tree::N)
        .def_readwrite("M", &Tree::M)
        .def_readwrite("dimx", &Tree::dimx)
        .def_readwrite("nsteps", &Tree::nsteps)
        .def("reset", &Tree::reset, py::call_guard<py::gil_scoped_release>())
        .def("init", &Tree::init, py::call_guard<py::gil_scoped_release>())
        .def("insert", &Tree::insert, py::call_guard<py::gil_scoped_release>())
        .def("prune", &Tree::prune, py::call_guard<py::gil_scoped_release>())
        .def("update", &Tree::update, py::call_guard<py::gil_scoped_release>())
        .def("get_path", &Tree::get_path, py::call_guard<py::gil_scoped_release>())
        .def("retrieve_xgeneration",&Tree::retrieve_xgeneration, py::call_guard<py::gil_scoped_release>())
        .def("bulk_load", &Tree::bulk_load, py::call_guard<py::gil_scoped_release>())
        .def("reconstruct", &Tree::reconstruct, py::call_guard<py::gil_scoped_release>())
        // This creates a brand new Tree object that is an exact clone of the current one
        .def("copy", [](const Tree &t) { return new Tree(t); });
}

