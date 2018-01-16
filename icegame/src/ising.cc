#include "ising.hpp"
#include <math.h>

IsingGame::IsingGame(INFO info) : sim_info(info) {
    model.init(sim_info);
    latt.init(sim_info);
    ising_config.init(sim_info);

    // pre-determined parameters
    N = sim_info.Num_sites;
    L = sim_info.lattice_size;

    h1_t = 0.0;
    h2_t = 0.0;
    h3_t = 0.0;
    J1 = 1.0;
    kT = 2.0;
    mag_fields.push_back(h1_t);
    mag_fields.push_back(h2_t);
    mag_fields.push_back(h3_t);
}

void IsingGame::InitModel() {
    model.set_J1(J1);
	model.initialization(&ising_config, &latt, 1);
	model.initialize_observable(&ising_config, &latt, kT, mag_fields);
    std::cout << "[GAME] All physical parameters are initialized\n";
}

void IsingGame::SetTemperature(double T){
    if (T >= 0.0)
        kT = T;
    std::cout << "[GAME] Set temperature kT = " << kT << "\n";
    //MCRun(1000);
}

void IsingGame::MCRun(int mcSteps) {
    tt.timer_begin();
    for (int i = 0; i < mcSteps; ++i) {
        model.MCstep(&ising_config, &latt);
    }
    tt.timer_end();

    std::cout << "[GAME] Monte Carlo runs " 
                << mcSteps << " steps with "
                << tt.timer_duration() << " seconds.\n"; 
    
    // Check whether it is icestates
    double Etot = model.total_energy(&ising_config, &latt);
    std::cout << "[GAME] Total Energy E = " << Etot << "\n";

    // Get the ising variables
    state_0 = ising_config.Ising;
    state_t = ising_config.Ising;
    state_tp1 = ising_config.Ising;

    // config_mean = _cal_mean(state_0);
    // config_stdev = _cal_stdev(state_0);

    //std::cout << "[GAME] Average Energy E = " << _cal_energy_of_state(state_0) << "\n";
    //std::cout << "[GAME] Defect Density D = " << _cal_defect_density_of_state(state_0) << "\n";
    //std::cout << "[GAME] Config mean = " << config_mean << " , and std = " << config_stdev << "\n";
}