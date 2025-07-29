# Pi3_SLAM
Prototype of a SLAM system build on top of π^3. 

Just stumbled upon π^3 and found it to be really robust and fast! 
Just so much fun nowadays to build a dense VO system in a day using a coding agent and 👌😁 where most of the magic happens in a **model(images)** call...

Works for my forward facing GoPro odometry videos but not as robust for general trajectories yet. Probably due to the very simple Sim3 estimator. Examples will follow...

## ToDo

 - [] integrate Open3D robust ICP with scaling=True
 - [] Test SL(4) solver from vggt-slam
 - [] Upload GoPro samples
 - [] Eval on Euroc
 - [] Loop closures, but not that relevant
 - [] Localize a second camera to the reconstructed one
 - [] scale using GPS priors
 - [] align to Gravity with Imu priors
 

