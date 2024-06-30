import os

from dotenv import load_dotenv
import xarray as xr

load_dotenv()
DATA = os.getenv("CACHE")
RAW_DATA_PATH = '/data/shared/brainio/brain-score/assy_dicarlo_MajajHong2015_public.nc' #where raw data is downloaded 
PROCESSED_DATA = os.path.join(DATA,'majajhong')

# get shared stimulus ids
regions = ['IT','V4']
subjects = ['Tito','Chabo']

# saving each region from each subject separately 
for subject in subjects:
    for region in regions:
        
        da = xr.open_dataset(RAW_DATA_PATH)
        # get data for shared images
        da = da.where(da.animal.isin(subject),drop=True)

        # get region's voxels
        da_region = da.where((da.region == region), drop=True)

        l = list(da_region.coords)
        # remove all other regions
        l.remove('image_id') # keep stimulus id
        da_region = da_region.drop(l) # drop other coords
        # get the average voxel response per image for region's voxels
        da_region = da_region.groupby('image_id').mean()
        da_region = da_region.rename({'image_id':'stimulus_id',
                                      'dicarlo.MajajHong2015.public':'x'})
        da_region.to_netcdf(os.path.join(PROCESSED_DATA,f'SUBJECT_{subject}_REGION_{region}'))