#include <iostream>
#include </Users/guan/opt/Eigen/Dense>
//#include </Users/fresh/opt/Eigen/Sparse>
#include <fstream>
#include <string>
//#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
//using Eigen::MatrixXd;


#define MAX(a,b) ((a) > (b))?(a):(b)

//typedef SparseMatrix<double, Eigen::RowMajor> Matrix_t;
typedef MatrixXd Matrix_t;

void loadMatrix(const char * file_name, Matrix_t & sm1)
{
    //fstream ifs("data");
    fstream ifs(file_name);

    string line;
    int mi = 0;
    int mj = 0;

    while (getline(ifs, line))
    {
        const char * pstr = line.c_str();

        mj = 0;
        while(1)
        {
            int pos = strspn(pstr, ".-0123456789");
            if (pos == 0)
                break;

            int val = atoi(pstr);
            //sm1.insert(mi, mj) = val;
            printf("%d, %d\n", mi, mj);
            sm1(mi, mj) = val;

            pstr += pos + 1;
            ++mj;
        }

        ++mi;
        fprintf(stderr, "<= %d\n", mi);

    }

    ifs.close();

}

int main()
{
    // Matrix_t sm1(39774, 6714);
   Matrix_t sm1(3, 3);

    fprintf(stderr, "> to load..\n");
    //loadMatrix("data", sm1);
    // loadMatrix("./train_mtx.csv", sm1);
    loadMatrix("./pca_test.csv", sm1);
    fprintf(stderr, "> to load.. ok\n");
//    cout << sm1 << endl;


    Matrix_t pca_conv;
    fprintf(stderr, "> to cal pca conv\n");
    {
        Matrix_t & train_data = sm1;
        Matrix_t B = train_data;

        VectorXd fmean = train_data.colwise().mean();

        for (int i = 0; i < B.rows(); ++i)
        {
            B.row(i) -= fmean;
        }


        fprintf(stderr, "> to mul adj\n");
        pca_conv = B.adjoint()*B;
        pca_conv = pca_conv / (train_data.rows() - 1);
    }
    fprintf(stderr, "> cal pca convok\n");

    fprintf(stderr, "> to cal pca eig ..\n");

    // cout << "conv " << endl;
    // cout << pca_conv << endl;

    Eigen::SelfAdjointEigenSolver<Matrix_t> eig(pca_conv);
    printf("cal pca eig ok\n\n");

    //VectorXf normalizedEigenValues =  eig.eigenvalues() / eig.eigenvalues().sum();
    VectorXd eig_values =  eig.eigenvalues();

    cout << "*** eig vectors ***" << endl;
    cout << eig.eigenvectors() << endl;
    cout << "*** eig values ***" << endl;
    cout << eig_values << endl;
    //cout << pca_conv;
#if 0
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
#endif
}
