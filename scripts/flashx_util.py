import argparse
import glob
import os
import json

from joblib import Parallel, delayed
import h5py
import numpy as np
import boxkit

class FlashxLoader:
    def __init__(self, directory):
        filenames = glob.glob(directory + '/*')
        chk_files = sorted([f for f in filenames if 'chk' in f])
        heater = h5py.File([f for f in filenames if 'htr' in f][0])

        frame = h5py.File(chk_files[0], 'r')
        string_runtime_params = {x[0].decode('UTF-8').strip(): x[1].decode('UTF-8').strip() for x in frame['string runtime parameters'][()]}
        real_runtime_params = {x[0].decode('UTF-8').strip(): x[1] for x in frame['real runtime parameters'][()]}
    
        integer_runtime_params = {x[0].decode('UTF-8').strip(): x[1] for x in frame['integer runtime parameters'][()]}
        nblocky, nblockx = int(integer_runtime_params['nblocky']), int(integer_runtime_params['nblockx'])
        
        integer_scalars = {x[0].decode('UTF-8').strip(): x[1] for x in frame['integer scalars'][()]}
        ny_block, nx_block = integer_scalars['nyb'], integer_scalars['nxb']
        
        xmax, xmin = frame['bounding box'][()][:,0,:].max(), frame['bounding box'][()][:,0,:].min()
        ymax, ymin = frame['bounding box'][()][:,1,:].max(), frame['bounding box'][()][:,1,:].min()

        Ny, Nx = nblocky*ny_block, nblockx*nx_block
        dx = (xmax - xmin)/Nx
        dy = (ymax - ymin)/Ny
        x_faces = np.linspace(xmin, xmax, Nx+1)
        y_faces = np.linspace(ymin, ymax, Ny+1)
        x_centers = 0.5*(x_faces[1:] + x_faces[:-1])
        y_centers = 0.5*(y_faces[1:] + y_faces[:-1])
       
        self.load_vars = ['dfun', 'temp', 'velx', 'vely', 'nrmx',
                            'nrmy', 'mflx', 'dust', 'rhoc', 'pres',
                            'fcx8', 'fcy8']
        self.save_vars = ['dfun', 'temperature', 'velx', 'vely', 'normx', 
                            'normy', 'massflux', 'divergence', 'density', 'pressure',
                            'velfacex', 'velfacey']

        self.data = {
                'x_centers': x_centers,
                'y_centers': y_centers,
                'x_faces': x_faces,
                'y_faces': y_faces,
                #'dfun': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'temperature': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'velx': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'vely': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'normx': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'normy': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'massflux': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'divergence': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'density': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'pressure': np.zeros((len(chk_files), Ny, Nx), dtype=np.float32),
                #'velfacex': np.zeros((len(chk_files), Ny, Nx+1), dtype=np.float32),
                #'velfacey': np.zeros((len(chk_files), Ny+1, Nx), dtype=np.float32),
                }

        self.parameters = {
                'geometry': string_runtime_params['geometry'],
                'xl_boundary_type': string_runtime_params['xl_boundary_type'],
                'xr_boundary_type': string_runtime_params['xr_boundary_type'],
                'yl_boundary_type': string_runtime_params['yl_boundary_type'],
                'yr_boundary_type': string_runtime_params['yr_boundary_type'],
                'num_blocks_x': nblockx,
                'num_blocks_y': nblocky,
                'nx_block': nx_block,
                'ny_block': ny_block,
                'dt': float(real_runtime_params['checkpointfileintervaltime']),
                't_initial': float(real_runtime_params['tinitial']),
                't_final': float(real_runtime_params['tmax']),
                'x_min': float(real_runtime_params['xmin']),
                'x_max': float(real_runtime_params['xmax']),
                'y_min': float(real_runtime_params['ymin']),
                'y_max': float(real_runtime_params['ymax']),
                'gravx': float(real_runtime_params['ins_gravx']),
                'gravy': float(real_runtime_params['ins_gravy']),
                'gravz': float(real_runtime_params['ins_gravz']),
                'prandtl': float(real_runtime_params['ht_prandtl']),
                'inv_reynolds': float(real_runtime_params['ins_invreynolds']),
                'inflow_velscale': float(real_runtime_params['ins_inflowvelscale']),
                'cpgas': float(real_runtime_params['mph_cpgas']),
                'mugas': float(real_runtime_params['mph_mugas']),
                'rhogas': float(real_runtime_params['mph_rhogas']), 
                'thcogas': float(real_runtime_params['mph_thcogas']), 
                'stefan': float(real_runtime_params['mph_stefan']),
                'heater': {k: heater['heater'][k][()].tolist()[0] for k in heater['heater'].keys()},
                'nuc_seed_radii': heater['init']['radii'][()].tolist(),
                'nuc_sites_x': heater['site']['x'][()].tolist(),
                'nuc_sites_y': heater['site']['y'][()].tolist(),
            }

        results = Parallel(n_jobs=-1)(delayed(self._load_data)(i, f) for i, f in enumerate(chk_files))

        for k in results[0].keys():
            self.data[k] = np.array([result[k] for result in results], dtype=np.float32)
            print(f'{k} of shape {self.data[k].shape}')


    def _load_data(self, time_index, filename):
        h5_frame = h5py.File(filename, 'r')
        frame = boxkit.read_dataset(filename, source='flash')
        blocks = frame.blocklist

        nblockx, nblocky = self.parameters['num_blocks_x'], self.parameters['num_blocks_y']
        
        x_bs, y_bs = self.parameters['ny_block'], self.parameters['nx_block']
        
        xmax, xmin = frame.xmax, frame.xmin 
        ymax, ymin = frame.ymax, frame.ymin

        Ny, Nx = nblocky*y_bs, nblockx*x_bs
        data_i = {k: np.zeros((Ny, Nx), dtype=np.float32) for k in self.save_vars if k not in ['velfacex', 'velfacey']}
        data_i['velfacex'] = np.zeros((Ny, Nx+1), dtype=np.float32)
        data_i['velfacey'] = np.zeros((Ny+1, Nx), dtype=np.float32)

        for load_var, save_var in zip(self.load_vars, self.save_vars):
            if load_var in frame.varlist:
                for block in blocks:
                    r = y_bs * round(int((Ny * (block.ymin - ymin))/(ymax - ymin))/y_bs)
                    c = x_bs * round(int((Nx * (block.xmin - xmin))/(xmax - xmin))/x_bs)
                    tmp = block[load_var]
                    data_i[save_var][r:r+y_bs, c:c+x_bs] = np.float32(tmp)
            else:
                for i, block in enumerate(h5_frame[load_var]):
                    block_xmin = h5_frame['bounding box'][()][i,0,:].min()
                    block_ymin = h5_frame['bounding box'][()][i,1,:].min()
                    r = y_bs * round(int((Ny * (block_ymin - ymin))/(ymax - ymin))/y_bs)
                    c = x_bs * round(int((Nx * (block_xmin - xmin))/(xmax - xmin))/x_bs)
                    if load_var == 'fcx8':
                        data_i[save_var][r:r+y_bs, c:c+x_bs] = np.float32(block[0,:,:-1])
                    elif load_var == 'fcy8':
                        data_i[save_var][r:r+y_bs, c:c+x_bs] = np.float32(block[0,:-1,:])
                    else:
                        self.data[save_var][r:r+y_bs, c:c+x_bs] = np.float32(block[0])
        data_i['velfacex'][:, -1] = 2 * data_i['velx'][:, -1] - data_i['velfacex'][:, -2]
        data_i['velfacey'][-1, :] = 2 * data_i['vely'][-1, :] - data_i['velfacey'][-2, :]

        return data_i

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_dir', type=str, help='Directory to load flashx simulation data from')
    parser.add_argument('--output_dir', type=str, help='Directory to save output visualization and unblocked hdf5 file')

    args = parser.parse_args()

    sims = [d for d in os.listdir(args.sim_dir) if 'Twall' in d]
    params_obj = []
    for sim in sims:
        print(f'Processing {sim}')
        sim_obj = FlashxLoader(os.path.join(args.sim_dir, sim))
        with h5py.File(os.path.join(args.output_dir, sim + '.hdf5'), 'w') as f:
            for key in sim_obj.data.keys():
                f.create_dataset(key, data=sim_obj.data[key])
        params_obj.append(sim_obj.parameters)
        print(f'Wrote {sim} to {args.output_dir}')

    import json
    json_path = os.path.join(args.output_dir, 'parameters.json')
    with open(json_path, 'w') as f_json:
        json.dump(params_obj, f_json, indent=4, default=str)
    print(f"✅ Processing complete.")


