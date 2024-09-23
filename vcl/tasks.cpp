#include "Labs/4-Animation/tasks.h"
#include "CustomFunc.inl"
#include "IKSystem.h"
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <spdlog/spdlog.h>
using namespace std;
using namespace chrono;

namespace VCX::Labs::Animation {
    void ForwardKinematics(IKSystem & ik, int StartIndex) {
        if (StartIndex == 0) {
            ik.JointGlobalRotation[0] = ik.JointLocalRotation[0];
            ik.JointGlobalPosition[0] = ik.JointLocalOffset[0];
            StartIndex                = 1;
        }

        for (int i = StartIndex; i < ik.JointLocalOffset.size(); i++) {
            // your code here: forward kinematics
            ik.JointGlobalRotation[i] = glm::normalize(ik.JointGlobalRotation[i - 1] * ik.JointLocalRotation[i]);
            ik.JointGlobalPosition[i] = ik.JointGlobalPosition[i - 1] + glm::rotate(ik.JointGlobalRotation[i - 1], ik.JointLocalOffset[i]);
        }
    }

    void InverseKinematicsCCD(IKSystem & ik, const glm::vec3 & EndPosition, int maxCCDIKIteration, float eps) {
        ForwardKinematics(ik, 0);
        // These functions will be useful: glm::normalize, glm::rotation, glm::quat * glm::quat
        for (int CCDIKIteration = 0; CCDIKIteration < maxCCDIKIteration && glm::l2Norm(ik.EndEffectorPosition() - EndPosition) > eps; CCDIKIteration++) {
            // your code here: ccd ik
            for (int i = ik.JointLocalOffset.size() - 2; i >= 0; i--) {
                glm::vec3 a                  = glm::normalize(ik.JointGlobalPosition[ik.JointLocalOffset.size() - 1] - ik.JointGlobalPosition[i]);
                glm::vec3 b                  = glm::normalize(EndPosition - ik.JointGlobalPosition[i]);
                ik.JointGlobalRotation[i]    = glm::rotation(a, b) * ik.JointGlobalRotation[i];
                ik.JointLocalRotation[i + 1] = glm::inverse(ik.JointGlobalRotation[i]) * ik.JointGlobalRotation[i + 1];
                ForwardKinematics(ik, i + 1);
            }
            ik.JointLocalRotation[0] = ik.JointGlobalRotation[0];
            ForwardKinematics(ik, 0);
        }
    }

    void InverseKinematicsFABR(IKSystem & ik, const glm::vec3 & EndPosition, int maxFABRIKIteration, float eps) {
        ForwardKinematics(ik, 0);
        int                    nJoints = ik.NumJoints();
        std::vector<glm::vec3> backward_positions(nJoints, glm::vec3(0, 0, 0)), forward_positions(nJoints, glm::vec3(0, 0, 0));
        for (int IKIteration = 0; IKIteration < maxFABRIKIteration && glm::l2Norm(ik.EndEffectorPosition() - EndPosition) > eps; IKIteration++) {
            // task: fabr ik
            // backward update
            glm::vec3 next_position         = EndPosition;
            backward_positions[nJoints - 1] = EndPosition;

            for (int i = nJoints - 2; i >= 0; i--) {
                // your code here
                backward_positions[i] = backward_positions[i + 1] - ik.JointOffsetLength[i + 1] * glm::normalize(backward_positions[i + 1] - ik.JointGlobalPosition[i]);
            }

            // forward update
            glm::vec3 now_position = ik.JointGlobalPosition[0];
            forward_positions[0]   = ik.JointGlobalPosition[0];
            for (int i = 0; i < nJoints - 1; i++) {
                // your code here
                forward_positions[i + 1] = forward_positions[i] + ik.JointOffsetLength[i + 1] * glm::normalize(backward_positions[i + 1] - forward_positions[i]);
            }
            ik.JointGlobalPosition = forward_positions; // copy forward positions to joint_positions
        }

        // Compute joint rotation by position here.
        for (int i = 0; i < nJoints - 1; i++) {
            ik.JointGlobalRotation[i] = glm::rotation(glm::normalize(ik.JointLocalOffset[i + 1]), glm::normalize(ik.JointGlobalPosition[i + 1] - ik.JointGlobalPosition[i]));
        }
        ik.JointLocalRotation[0] = ik.JointGlobalRotation[0];
        for (int i = 1; i < nJoints - 1; i++) {
            ik.JointLocalRotation[i] = glm::inverse(ik.JointGlobalRotation[i - 1]) * ik.JointGlobalRotation[i];
        }
        ForwardKinematics(ik, 0);
    }

    IKSystem::Vec3ArrPtr IKSystem::BuildCustomTargetPosition() {
        // get function from https://www.wolframalpha.com/input/?i=Albert+Einstein+curve
        int nums      = 5000;
        using Vec3Arr = std::vector<glm::vec3>;
        std::shared_ptr<Vec3Arr> custom(new Vec3Arr(nums));
        int                      index = 0;
        for (int i = 0; i < nums; i++) {
            float x_val = 1.5e-3f * custom_x(92 * glm::pi<float>() * i / nums);
            float y_val = 1.5e-3f * custom_y(92 * glm::pi<float>() * i / nums);
            if (std::abs(x_val) < 1e-3 || std::abs(y_val) < 1e-3) continue;
            (*custom)[index++] = glm::vec3(1.6f - x_val, 0.0f, y_val - 0.2f);
        }
        custom->resize(index);
        return custom;
    }

    void AdvanceMassSpringSystem(MassSpringSystem & system, float dt, Eigen::VectorXf & prevState, int Iteration) {
        int n                                                      = system.Positions.size(); // point num
        int m                                                      = system.Springs.size();   // spring num
        //Iteration                                                  = 100;
        int const                                        w         = 1e10;
        Eigen::VectorXf                                  currState = Eigen::VectorXf::Zero(3 * n);
        Eigen::SparseMatrix<float>                       L(3 * n, 3 * n);
        Eigen::SparseMatrix<float>                       M(3 * n, 3 * n);
        Eigen::SparseMatrix<float>                       J(3 * n, 3 * m);
        Eigen::VectorXf                                  f_ext = Eigen::VectorXf::Zero(3 * n);
        Eigen::VectorXf                                  b     = Eigen::VectorXf::Zero(3 * n);
        Eigen::VectorXf                                  d     = Eigen::VectorXf::Zero(3 * m);
        Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> cholesky;

        for (int i = 0; i < n; i++) {
            f_ext[i * 3 + 1] = -system.Mass * system.Gravity;
        }

        std::vector<Eigen::Triplet<float>> coefficients;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 3; j++) {
                coefficients.push_back(Eigen::Triplet<float>(3 * i + j, 3 * i + j, system.Mass));
            }
        }
        M.setFromTriplets(coefficients.begin(), coefficients.end());

        std::vector<Eigen::Triplet<float>> LTriplets;
        for (int i = 0; i < m; i++) {
            int   si = system.Springs[i].AdjIdx.first;
            int   ei = system.Springs[i].AdjIdx.second;
            float k  = system.Stiffness;
            for (int j = 0; j < 3; j++) {
                LTriplets.push_back(Eigen::Triplet<float>(3 * si + j, 3 * si + j, k));
                LTriplets.push_back(Eigen::Triplet<float>(3 * si + j, 3 * ei + j, -k));
                LTriplets.push_back(Eigen::Triplet<float>(3 * ei + j, 3 * ei + j, k));
                LTriplets.push_back(Eigen::Triplet<float>(3 * ei + j, 3 * si + j, -k));
            }
        }
        L.setFromTriplets(LTriplets.begin(), LTriplets.end());

        std::vector<Eigen::Triplet<float>> JTriplets;
        for (int i = 0; i < m; i++) {
            int   si = system.Springs[i].AdjIdx.first;
            int   ei = system.Springs[i].AdjIdx.second;
            float k  = system.Stiffness;

            for (int j = 0; j < 3; j++) {
                JTriplets.push_back(Eigen::Triplet<float>(3 * si + j, 3 * i + j, -k));
                JTriplets.push_back(Eigen::Triplet<float>(3 * ei + j, 3 * i + j, k));
            }
        }
        J.setFromTriplets(JTriplets.begin(), JTriplets.end());

        float                              h2 = (dt) * (dt);
        Eigen::SparseMatrix<float>         Q  = M + h2 * L;
        Eigen::SparseMatrix<float>         Qp(3 * n, 3 * n);
        std::vector<Eigen::Triplet<float>> QTriplets;
        for (int i = 0; i < n; i++) {
            if (system.Fixed[i]) {
                for (int j = 0; j < 3; j++) {
                    QTriplets.push_back(Eigen::Triplet<float>(3 * i + j, 3 * i + j, w));
                }
            }
        }
        Qp.setFromTriplets(QTriplets.begin(), QTriplets.end());
        Q += Qp;
        cholesky.compute(Q);
        for (int i = 0; i < n; i++) {
            currState[3 * i]     = system.Positions[i].x;
            currState[3 * i + 1] = system.Positions[i].y;
            currState[3 * i + 2] = system.Positions[i].z;
        }
        Eigen::VectorXf y = M * ((1 + system.Alpha) * currState - system.Alpha * prevState);
        prevState         = currState;
        auto start = system_clock::now();
        for (int pp = 0; pp < Iteration; pp++) {
            for (int i = 0; i < m; i++) {
                int       si = system.Springs[i].AdjIdx.first;
                int       ei = system.Springs[i].AdjIdx.second;
                glm::vec3 tp = glm::vec3(currState[3 * ei], currState[3 * ei + 1], currState[3 * ei + 2]) - glm::vec3(currState[3 * si], currState[3 * si + 1], currState[3 * si + 2]);
                tp           = glm::normalize(tp) * system.Springs[i].RestLength;
                d[3 * i]     = tp.x;
                d[3 * i + 1] = tp.y;
                d[3 * i + 2] = tp.z;
            }
            b = h2 * J * d + y + h2 * f_ext;
            for (std::size_t i = 0; i < system.Positions.size(); i++) {
                if (system.Fixed[i]) {
                    b[3 * i] += w * currState[3 * i];
                    b[3 * i + 1] += w * currState[3 * i + 1];
                    b[3 * i + 2] += w * currState[3 * i + 2];
                }
            }
            currState = cholesky.solve(b);
        }
        auto end      = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        //cout << "»¨·ÑÁË"
        //     << double(duration.count()) * microseconds::period::num / microseconds::period::den
        //     << "Ãë" << endl;
        for (std::size_t i = 0; i < system.Positions.size(); i++) {
            if (system.Fixed[i])
                continue;

            system.Velocities[i] = (glm::vec3(currState[3 * i], currState[3 * i + 1], currState[3 * i + 2]) - system.Positions[i]) / dt;

            system.Positions[i] = glm::vec3(currState[3 * i], currState[3 * i + 1], currState[3 * i + 2]);
        }

    }
} // namespace VCX::Labs::Animation
