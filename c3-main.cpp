
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>

#include <carla/client/Vehicle.h>

//pcl code
//#include "render/render.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include "helper.h"
#include <sstream>
#include <chrono> 
#include <ctime> 
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h>   // TicToc

PointCloudT pclCloud;
cc::Vehicle::Control control;
std::chrono::time_point<std::chrono::system_clock> currentTime;
vector<ControlState> cs;

bool refresh_view = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
{

  	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
	if (event.getKeySym() == "Right" && event.keyDown()){
		cs.push_back(ControlState(0, -0.02, 0));
  	}
	else if (event.getKeySym() == "Left" && event.keyDown()){
		cs.push_back(ControlState(0, 0.02, 0)); 
  	}
  	if (event.getKeySym() == "Up" && event.keyDown()){
		cs.push_back(ControlState(0.1, 0, 0));
  	}
	else if (event.getKeySym() == "Down" && event.keyDown()){
		cs.push_back(ControlState(-0.1, 0, 0)); 
  	}
	if(event.getKeySym() == "a" && event.keyDown()){
		refresh_view = true;
	}
}

void Accuate(ControlState response, cc::Vehicle::Control& state){

	if(response.t > 0){
		if(!state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = false;
			state.throttle = min(response.t, 1.0f);
		}
	}
	else if(response.t < 0){
		response.t = -response.t;
		if(state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = true;
			state.throttle = min(response.t, 1.0f);

		}
	}
	state.steer = min( max(state.steer+response.s, -1.0f), 1.0f);
	state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer){

	BoxQ box;
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    box.cube_length = 4;
    box.cube_width = 2;
    box.cube_height = 2;
	renderBox(viewer, box, num, color, alpha);
}


/**
 * @brief Perform Iterative Closest Point (ICP) algorithm to align the source point cloud to the target point cloud.
 * 
 * This function uses the ICP algorithm to find the best alignment between the source and target point clouds.
 * It initializes the source point cloud with a given starting pose, applies the ICP algorithm, and returns the
 * transformation matrix that aligns the source to the target.
 * 
 * @param target The target point cloud to which the source point cloud will be aligned.
 * @param source The source point cloud that will be aligned to the target.
 * @param startingPose The initial pose of the source point cloud.
 * @return Eigen::Matrix4d The transformation matrix that aligns the source point cloud to the target point cloud.
 */
Eigen::Matrix4d performICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose) {
 
    // Initialize the transformation matrix to identity
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
 
    // Align the source point cloud with the starting pose
    Eigen::Matrix4d initTransform = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, startingPose.position.x, startingPose.position.y, startingPose.position.z);
    PointCloudT::Ptr transformSource (new PointCloudT);
    pcl::transformPointCloud(*source, *transformSource, initTransform);
 
    // Start the timer
    pcl::console::TicToc time;
    time.tic();
    
    // Set up the ICP algorithm
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(8); // Set the maximum number of iterations
    icp.setInputSource(transformSource); // Set the source point cloud
    icp.setInputTarget(target); // Set the target point cloud
    icp.setMaxCorrespondenceDistance(2.0); // Set the maximum correspondence distance
    icp.setTransformationEpsilon(1e-4); // Set the transformation epsilon (convergence criteria)
    icp.setRANSACOutlierRejectionThreshold(10); // Set the RANSAC outlier rejection threshold
 
    // Perform the ICP alignment
    PointCloudT::Ptr cloud_icp (new PointCloudT); // ICP output point cloud
    icp.align(*cloud_icp);
 
    // Check if the ICP algorithm has converged
    if (icp.hasConverged()) {
        // Get the final transformation matrix
        transformation_matrix = icp.getFinalTransformation().cast<double>();
        // Combine with the initial transformation
        transformation_matrix = transformation_matrix * initTransform;
        // Print the transformation matrix
        print4x4Matrix(transformation_matrix);
 
        return transformation_matrix;
    } else {
        // Print a warning if ICP did not converge
        std::cout << "WARNING: ICP did not converge" << std::endl;
    }
    
    return transformation_matrix;
}

/**
 * @brief Perform Normal Distributions Transform (NDT) algorithm to align the target point cloud to the initial pose.
 * 
 * This function uses the NDT algorithm to find the best alignment between the target point cloud and an initial pose.
 * It initializes the target point cloud with a given starting pose, applies the NDT algorithm, and returns the
 * transformation matrix that aligns the target to the initial pose.
 * 
 * @param ndt The NDT object configured with the necessary parameters.
 * @param target The target point cloud that will be aligned to the initial pose.
 * @param startingPose The initial pose of the target point cloud.
 * @return Eigen::Matrix4d The transformation matrix that aligns the target point cloud to the initial pose.
 */
Eigen::Matrix4d performNDT(
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt, 
    PointCloudT::Ptr target, 
    Pose startingPose) {
 
    // Initialize the transformation matrix to identity
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

    // Create the initial guess transformation matrix from the starting pose
    Eigen::Matrix4f init_guess = transform3D(
        startingPose.rotation.yaw, 
        startingPose.rotation.pitch, 
        startingPose.rotation.roll, 
        startingPose.position.x, 
        startingPose.position.y, 
        startingPose.position.z
    ).cast<float>();

    // Start the timer
    pcl::console::TicToc time;
    time.tic();
 
    // Set the maximum number of registration iterations
    ndt.setMaximumIterations(8);
    ndt.setInputSource(target); // Set the target point cloud
     
    // Perform the NDT alignment
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ndt(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*cloud_ndt, init_guess);
 
    // Check if the NDT algorithm has converged
    if (ndt.hasConverged()) {
        // Get the final transformation matrix
        transformation_matrix = ndt.getFinalTransformation().cast<double>();
    }
 
    return transformation_matrix;
}

int main(){

	auto client = cc::Client("localhost", 2000);
	client.SetTimeout(2s);
	auto world = client.GetWorld();

	auto blueprint_library = world.GetBlueprintLibrary();
	auto vehicles = blueprint_library->Filter("vehicle");

	auto map = world.GetMap();
	auto transform = map->GetRecommendedSpawnPoints()[1];
	auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

	//Create lidar
	auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
	// CANDO: Can modify lidar values to get different scan resolutions
	lidar_bp.SetAttribute("upper_fov", "15");
    lidar_bp.SetAttribute("lower_fov", "-25");
    lidar_bp.SetAttribute("channels", "32");
    lidar_bp.SetAttribute("range", "30");
	lidar_bp.SetAttribute("rotation_frequency", "60");
	lidar_bp.SetAttribute("points_per_second", "500000");

	auto user_offset = cg::Location(0, 0, 0);
	auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
	auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
	auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
	bool new_scan = true;
	std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	viewer->setBackgroundColor (0, 0, 0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
	Pose pose(Point(0,0,0), Rotate(0,0,0));

	// Load map
	PointCloudT::Ptr mapCloud(new PointCloudT);
  	pcl::io::loadPCDFile("map.pcd", *mapCloud);
  	cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	renderPointCloud(viewer, mapCloud, "map", Color(0,0,1)); 

	typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr scanCloud (new pcl::PointCloud<PointT>);

	lidar->Listen([&new_scan, &lastScanTime, &scanCloud](auto data){

		if(new_scan){
			auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
			for (auto detection : *scan){
				if((detection.x*detection.x + detection.y*detection.y + detection.z*detection.z) > 8.0){
					pclCloud.points.push_back(PointT(detection.x, detection.y, detection.z));
				}
			}
			if(pclCloud.points.size() > 5000){ // CANDO: Can modify this value to get different scan resolutions
				lastScanTime = std::chrono::system_clock::now();
				*scanCloud = pclCloud;
				new_scan = false;
			}
		}
	});
	
	Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));
	double maxError = 0;
	bool scanInit = true;
	bool useICP = false;

	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	// Set the NDT termination criteria
	// Note: we set maximum iterations inside the `NDT` function itself
	// ndt.setMaximumIterations(kMaximumIterationsNDT);
	ndt.setTransformationEpsilon(1e-4);
	// Set the NDT hyperparameters
	ndt.setStepSize(0.1);
	ndt.setResolution(1.0);
	// Set the fixed input point cloud
	ndt.setInputTarget(mapCloud);


	while (!viewer->wasStopped())
  	{
		while(new_scan){
			std::this_thread::sleep_for(0.1s);
			world.Tick(1s);
		}
		if(refresh_view){
			viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x+1, pose.position.y+1, 0, 0, 0, 1);
			refresh_view = false;
		}
		
		viewer->removeShape("box0");
		viewer->removeShape("boxFill0");
		Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;
		drawCar(truePose, 0,  Color(1,0,0), 0.7, viewer);
		double theta = truePose.rotation.yaw;
		double stheta = control.steer * pi/4 + theta;
		viewer->removeShape("steer");
		renderRay(viewer, Point(truePose.position.x+2*cos(theta), truePose.position.y+2*sin(theta),truePose.position.z),  Point(truePose.position.x+4*cos(stheta), truePose.position.y+4*sin(stheta),truePose.position.z), "steer", Color(0,1,0));


		ControlState accuate(0, 0, 1);
		if(cs.size() > 0){
			accuate = cs.back();
			cs.clear();

			Accuate(accuate, control);
			vehicle->ApplyControl(control);
		}

  		viewer->spinOnce ();
		
		if(!new_scan){
			new_scan = true;

			if(scanInit) {
				pose = truePose;
				scanInit = false;
			}

			// TODO: (Filter scan using voxel filter)
			pcl::VoxelGrid<PointT> vg;
			vg.setInputCloud(scanCloud);
			double filterRes = 0.05;
			vg.setLeafSize(filterRes, filterRes, filterRes);
			vg.setMinimumPointsNumberPerVoxel(1);
			typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
			vg.filter(*cloudFiltered);

			pose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;

			// TODO: Find pose transform by using ICP or NDT matching
			Eigen::Matrix4d transformedMatrix;
			if (useICP) {
				transformedMatrix = performICP(mapCloud, cloudFiltered, pose);
			} else {
				transformedMatrix = performNDT(ndt, cloudFiltered, pose);
			}
			
			pose = getPose(transformedMatrix);

			// TODO: Transform scan so it aligns with ego's actual pose and render that scan
			PointCloudT::Ptr transformedScan (new PointCloudT);
			pcl::transformPointCloud (*cloudFiltered, *transformedScan, transformedMatrix);

			viewer->removePointCloud("scan");
			// TODO: Change `scanCloud` below to your transformed scan
			renderPointCloud(viewer, transformedScan, "scan", Color(1,0,0) );

			viewer->removeAllShapes();
			drawCar(pose, 1,  Color(0,1,0), 0.35, viewer);
          
          	double poseError = sqrt( (truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y) );
			if(poseError > maxError)
				maxError = poseError;
			double distDriven = sqrt( (truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y) );
			viewer->removeShape("maxE");
			viewer->addText("Max Error: "+to_string(maxError)+" m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE",0);
			viewer->removeShape("derror");
			viewer->addText("Pose error: "+to_string(poseError)+" m", 200, 150, 32, 1.0, 1.0, 1.0, "derror",0);
			viewer->removeShape("dist");
			viewer->addText("Distance: "+to_string(distDriven)+" m", 200, 200, 32, 1.0, 1.0, 1.0, "dist",0);

			if(maxError > 1.2 || distDriven >= 170.0 ){
				viewer->removeShape("eval");
			if(maxError > 1.2){
				viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval",0);
			}
			else{
				viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval",0);
			}
		}

			pclCloud.points.clear();
		}
  	}
	return 0;
}

