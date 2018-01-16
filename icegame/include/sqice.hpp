/*
 * This program serves as new icegame, simplified versnio of environment for learning loop algorithm.
 */
#pragma once

// Switch of the debug mode.
// #define DEBUG

// cpp standard libs
#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <string>

// monte carlo libraries
#include "sample.hpp"
#include "hamiltonian.hpp"
#include "lattice.hpp"
#include "timer.hpp"

// boost.python intefaces
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <numpy/ndarrayobject.h> // ensure you include this header

//// Constants used in Icegame ////
const int NUM_OF_ACTIONS = 7;
const int METROPOLIS_PROPOSAL = 6;
const int NULL_SITE = -1;
const int SPIN_UP = 1;
const int SPIN_DOWN = -1;

const double SPIN_UP_SUBLATT_A = +0.75;
const double SPIN_UP_SUBLATT_B = +0.25;
const double SPIN_DOWN_SUBLATT_A = -0.75;
const double SPIN_DOWN_SUBLATT_B = -0.25;

const double DEFECT_MAP_DEFAULT_VALUE = 0.0;
const double ENERGY_MAP_DEFAULT_VALUE = 0.0;
const double EMPTY_MAP_VALUE = 0.0;
const double OCCUPIED_MAP_VALUE = 1.0;
const double AVERAGE_GORUND_STATE_ENERGY = -1.0;
const double AVERAGE_GORUND_STATE_DEFECT_DENSITY = 0.0;
const double ACCEPT_VALUE = 1.0;
const double REJECT_VALUE = -1.0;
const double DEFECT_DENSITY_THRESHOLD = 0.2;

const double AGENT_OCCUPIED_VALUE = 1.0;
//const double AGENT_OCCUPIED_SUBLATT_A = +1.0;
//const double AGENT_OCCUPIED_SUBLATT_B = -1.0;
//const double AGENT_FORESEE_VALUE = 0.75;
//const double AGENT_FORESEE_SUBLATT_A = +0.75;
//const double AGENT_FORESEE_SUBLATT_B = -0.75;

//// End of Constants ////

using namespace boost::python; 

// wrap c++ array as numpy array
static boost::python::object float_wrap(const std::vector<double> &vec) {
    npy_intp size = vec.size();
    double *data = const_cast<double *>(&vec[0]);

    npy_intp shape[1] = { size }; // array size
    PyObject* obj = PyArray_New(&PyArray_Type, 1, shape, NPY_DOUBLE, // data type
                                NULL, data, // data pointer
                                0, NPY_ARRAY_CARRAY, // NPY_ARRAY_CARRAY_RO for readonly
                                NULL);
    handle<> array( obj );
    return object(array);
}

// avoid using template
static boost::python::object int_wrap(const std::vector<int> &vec) {
    npy_intp size = vec.size();
    int *data = const_cast<int*>(&vec[0]);
    npy_intp shape[1] = { size }; // array size
    PyObject* obj = PyArray_New(&PyArray_Type, 1, shape, NPY_INT, // data type
                                NULL, data, // data pointer
                                0, NPY_ARRAY_CARRAY, // NPY_ARRAY_CARRAY_RO for readonly
                                NULL);
    handle<> array( obj );
    return object(array);
}

// conversion for std::vector to python list
template <class T>
struct Vec2List {
    static PyObject* convert (const std::vector<T> &vec) {
        boost::python::list *l = new boost::python::list();
        for (size_t i = 0; i < vec.size(); ++i)
            (*l).append(vec[i]);
        return l->ptr();
    }
};

/* Function Naming Convention:
    Upper Case function used for python interface
    lower case used as member functions
*/

enum DIR {RIGHT, DOWN, LEFT, UP, LOWER_NEXT, UPPER_NEXT};

class SQIceGame {
    public:
        // Init Constructor
        SQIceGame (INFO info);
        // Init all physical parameters
        void InitModel();
        void SetTemperature(double T);

        // Equalibirum
        void MCRun(int mcSteps);

        // Two kinds of action operation
        //    * directional action (up/down/left/right/upper_next/lower_next)
        //    * TODO coordinate action (depends on system size)
        //    return:
        //        vector of [accept, dE, dd, dC]
        //

        vector<double> Draw(int dir_idx);

        // Metropolis action
        vector<double> Metropolis();

        inline void FlipTrajectory() {flip_along_traj(agent_site_trajectory);};

        int Start(int init_site);
        int Restart(int init_site);
        int Reset(int init_site);

        void ClearBuffer();
        void UpdateConfig();

        inline int GetAgentSite() {return get_agent_site();};
        object GetStateTMap();
        object GetCanvasMap(); 
        object GetEnergyMap();
        object GetDefectMap();

        // New state tricks.
        object GetStateTMapColor();
        object GetStateDifferenceMap();
        object ValidActionMap();

        // Statistical Informations
        int GetStartPoint();
        inline unsigned long GetTotalSteps() {return num_total_steps;};
        inline unsigned long GetEpisode() {return num_episode;};
        inline int GetEpStepCounter() {return ep_step_counter;};
        // Number of successfully update the configuration
        inline unsigned long GetUpdatedCounter() {return updated_counter;};
        // Number of calling Metropolis
        inline unsigned long GetUpdatingCounter() {return num_updates;};
        inline double GetTotalAcceptanceRate() 
                {return updated_counter/(double) num_updates;};

        inline vector<int> GetAcceptedLen() {return accepted_looplength;};
        inline vector<int> GetTrajectory() {return agent_site_trajectory;};
        inline vector<int> GetActionStatistics() {return action_statistics;};
        inline vector<int> GetEpActionCounters() {return ep_action_counters;};
        inline vector<int> GetEpActionList() {return ep_action_list;};

        bool TimeOut();

        void update_ice_config();
        /// this function used fliiping ice config according to states?

        void flip_site(int site);
        void flip_agent_site();
        void flip_along_traj(const vector<int> &traj);

        // return new agent site
        int go(int dir);
        int how_to_go(int site);

        // propose a move satisfying the ice-rule
        // return a site
        int icemove(bool);

        void update_state_to_config();
        void restore_config_to_state();

        void clear_all();
        void clear_maps();
        void clear_counters();
        void clear_lists();

        int get_neighbor_site_by_direction(int dir);
        int get_direction_by_sites(int site, int next_site);
        
        vector<int> get_neighbor_sites();
        vector<int> get_neighbor_spins();
        vector<int> get_neighbor_candidates(bool same_spin);

        void set_agent_site(int site);

        /* get funcs */
        int  get_agent_site();
        int  get_agent_spin();
        int  get_spin(int site);

    private:
        // private functions
        double _cal_energy_of_state(const vector<int> &s);
        double _cal_energy_of_site(const vector<int> &s, int site);
        double _cal_defect_density_of_state(const vector<int> &s);
        int _cal_config_t_difference();
        int _count_config_difference(const vector<int> &c1, const vector<int> &c2);
        // magic function compute periodic boundary condition
        int inline _pdb(int site, int d, int l) {return ((site + d) % l + l) % l;};

        // Loop algorithms are empty functions
        void long_loop_algorithm();
        void short_loop_algorithm();

        void _print_vector(const vector<int> &v);
        double _cal_mean(const vector<int> &s);
        double _cal_stdev(const vector<int> &s);
        bool _is_visited(int site);
        bool _is_traj_continuous();
        bool _is_traj_intersect();
        bool _is_start_end_meets(int site);

        // Physical System
        INFO sim_info;
        Square latt;
        Square_ice model;
        Sample ice_config;

        double config_mean; 
        double config_stdev;

        unsigned int L, N;
        double kT;
        double h1_t, h2_t, h3_t;
        double J1;
        vector<double> mag_fields;

        // RL intefaces
        int agent_site;
        vector<int> agent_site_trajectory;
        vector<int> agent_spin_trajectory;

        /* Statistics of Game */

        unsigned long num_total_steps;
        unsigned long num_episode; // number of resets, TODO: Change point of views!

        unsigned long num_restarts;
        unsigned long num_resets;

        unsigned long num_updates; // number of calling Metropolis
        unsigned long updated_counter; // number of successfully updated
        int same_ep_counter; // records for playing the same game

        int update_interval; // 

        int init_agent_site;
        unsigned int ep_step_counter; // counts steps each episode
        vector<int> ep_site_counters;
        vector<int> ep_action_counters;
        vector<int> ep_action_list;
        vector<int> action_statistics;   // not reset
        vector<int> accepted_looplength; // not reset

        // MAPS
        /* Current states */ 
        vector<int> state_0;
        vector<int> state_t;
        vector<int> state_tp1;

        // maps always use double
        vector<double> agent_map;
        vector<double> canvas_traj_map;
        vector<double> canvas_spin_map;
        vector<double> energy_map;
        vector<double> defect_map;
        vector<int> diff_map;

        // utilities
        Timer tt;

};

BOOST_PYTHON_MODULE(icegame)
{
    class_<INFO>("INFO", init<int, int, int, int, int, int, int, int>())
    ;

    import_array();

    to_python_converter<std::vector<int, class std::allocator<int> >, Vec2List<int> >();
    to_python_converter<std::vector<double, class std::allocator<double> >, Vec2List<double> >();

    class_<SQIceGame>("SQIceGame", init<INFO>())
        .def("init_model", &SQIceGame::InitModel)
        .def("set_temperature", &SQIceGame::SetTemperature)
        .def("mc_run", &SQIceGame::MCRun)

        .def("start", &SQIceGame::Start)
        .def("restart", &SQIceGame::Restart)
        .def("reset", &SQIceGame::Reset)
        .def("timeout", &SQIceGame::TimeOut)

        .def("clear_buffer", &SQIceGame::ClearBuffer)

        // REVISE
        .def("draw", &SQIceGame::Draw)
        .def("metropolis", &SQIceGame::Metropolis)
        .def("flip_trajectory", &SQIceGame::FlipTrajectory)
        .def("update_config", &SQIceGame::UpdateConfig)

        .def("get_agent_site", &SQIceGame::GetAgentSite)
        .def("get_canvas_map", &SQIceGame::GetCanvasMap)
        .def("get_state_t_map", &SQIceGame::GetStateTMap)
        .def("get_state_t_map_color", &SQIceGame::GetStateTMapColor)
        .def("get_energy_map", &SQIceGame::GetEnergyMap)
        .def("get_defect_map", &SQIceGame::GetDefectMap)
        .def("get_valid_action_map", &SQIceGame::ValidActionMap)

        .def("get_start_point", &SQIceGame::GetStartPoint)
        .def("get_total_steps", &SQIceGame::GetTotalSteps)
        .def("get_episode", &SQIceGame::GetEpisode)
        .def("get_ep_step_counter", &SQIceGame::GetEpStepCounter)
        .def("get_total_acceptance_rate", &SQIceGame::GetTotalAcceptanceRate)
        .def("get_action_statistics", &SQIceGame::GetActionStatistics)
        .def("get_ep_action_counters", &SQIceGame::GetEpActionCounters)
        .def("get_ep_action_list", &SQIceGame::GetEpActionList)
        .def("get_updated_counter", &SQIceGame::GetUpdatedCounter)
        .def("get_updating_counter", &SQIceGame::GetUpdatingCounter)
        .def("get_accepted_length", &SQIceGame::GetAcceptedLen)
        .def("get_trajectory", &SQIceGame::GetTrajectory)
    ;
}
