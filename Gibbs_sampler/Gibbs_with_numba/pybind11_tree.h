#ifndef _INCL_PYBIND11_TREE_
#define _INCL_PYBIND11_TREE_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>   
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXi;

class Tree {
public:
    // --- THIS IS THE MISSING LINE ---
    // It ensures that when Python creates this object, it respects 
    // the strict memory alignment requirements of the Eigen library.
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int N;
    int M;
    int dimx;
    int nsteps;

    VectorXi a_star;
    VectorXi o_star;
    MatrixXd x_star;
    VectorXi l_star;

    Tree(int N, int M, int dimx);
    ~Tree();

    void reset();
    void init(const MatrixXd& x_0);
    void insert(const MatrixXd& x, const VectorXi& a);
    void prune(const VectorXi& o);
    void update(const MatrixXd& x, const VectorXi& a);
    
    // Your new bulk function
    void bulk_load(const MatrixXd& full_traj, const MatrixXd& full_ancestors);
    
    MatrixXd retrieve_xgeneration(int lag);
    void double_size();
    MatrixXd get_path(int n);
    void reconstruct(int N, int M, int d, int n,
                     const VectorXi& a, const VectorXi& o,
                     const MatrixXd& x, const VectorXi& l);
};

#endif