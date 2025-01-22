
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

// Normal Distributions Transform (NDT) Scan Matching Algorithm
Eigen::Matrix4d performNDTMatching(
    PointCloudT::Ptr target, PointCloudT::Ptr source, Pose initialPose,
    double epsilon = 1e-6, int maxIter = 60) {

    // Start timer to measure processing time
    pcl::console::TicToc timer;
    timer.tic();

    // Initialize the final transformation matrix as an identity matrix
    Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();

    // Align the source point cloud with the initial pose to get the initial guess for NDT
    Eigen::Matrix4f initialGuess = transform3D(
        initialPose.rotation.yaw, initialPose.rotation.pitch, initialPose.rotation.roll,
        initialPose.position.x, initialPose.position.y, initialPose.position.z
    ).cast<float>();

    // Initialize the Normal Distributions Transform (NDT)
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;

    // Set NDT parameters
    ndt.setTransformationEpsilon(epsilon); // Minimum transformation difference for termination
    ndt.setStepSize(0.1); // Maximum step size for More-Thuente line search
    ndt.setResolution(1.0); // Resolution of NDT grid structure
    ndt.setMaximumIterations(maxIter); // Maximum number of registration iterations

    // Set the point cloud to be aligned
    ndt.setInputSource(source);

    // Set the target point cloud to align the source cloud to
    ndt.setInputTarget(target);

    // Initialize the output point cloud after applying NDT
    PointCloudT::Ptr outputCloud(new PointCloudT);
    // Apply NDT to get the final transformation matrix
    ndt.align(*outputCloud, initialGuess);

    // Check if NDT has converged
    if (ndt.hasConverged()) {
        // Get the final transformation matrix
        transformationMatrix = ndt.getFinalTransformation().cast<double>();

        // Display convergence information
        std::cout << "NDT converged in " << timer.toc() << " ms with a fitness score of " << ndt.getFitnessScore() << std::endl;
        std::cout << "Returning final NDT transformation matrix!" << std::endl;
    } else {
        std::cout << "WARNING: NDT did not converge!" << std::endl;
        std::cout << "Returning identity matrix!" << std::endl;
    }

    // Return the final transformation matrix obtained by NDT
    return transformationMatrix;
}

// Iterative Closest Point (ICP) Algorithm
Eigen::Matrix4d performICPMatching(
    PointCloudT::Ptr target, PointCloudT::Ptr source, Pose initialPose,
    double epsilon = 1e-4, int maxIter = 16) {

    // Start timer to measure processing time
    pcl::console::TicToc timer;
    timer.tic();

    // Initialize the final transformation matrix as an identity matrix
    Eigen::Matrix4d transformationMatrix = Eigen::Matrix4d::Identity();

    // Align the source with the initial pose to get the initial transform
    Eigen::Matrix4d initialTransform = transform3D(
        initialPose.rotation.yaw, initialPose.rotation.pitch, initialPose.rotation.roll,
        initialPose.position.x, initialPose.position.y, initialPose.position.z
    );

    // Transform the source point cloud using the initial transform
    PointCloudT::Ptr transformedSource(new PointCloudT);
    pcl::transformPointCloud(*source, *transformedSource, initialTransform);

    // Initialize the Iterative Closest Point (ICP)
    pcl::IterativeClosestPoint<PointT, PointT> icp;

    // Set ICP parameters
    icp.setTransformationEpsilon(epsilon); // Minimum transformation difference for termination
    icp.setMaxCorrespondenceDistance(2.0); // Maximum correspondence distance
    icp.setRANSACOutlierRejectionThreshold(10); // RANSAC outlier rejection threshold
    icp.setMaximumIterations(maxIter); // Maximum number of registration iterations

    // Set the point cloud to be aligned
    icp.setInputSource(transformedSource);

    // Set the target point cloud to align the source cloud to
    icp.setInputTarget(target);

    // Initialize the output point cloud after applying ICP
    PointCloudT::Ptr outputCloud(new PointCloudT);
    // Apply ICP to get the final transformation matrix
    icp.align(*outputCloud);

    // Check if ICP has converged
    if (icp.hasConverged()) {
        // Get the final transformation matrix
        transformationMatrix = icp.getFinalTransformation().cast<double>();
        // Apply the initial transform to the final transform
        transformationMatrix = (transformationMatrix * initialTransform).cast<double>();

        // Display convergence information
        std::cout << "ICP converged in " << timer.toc() << " ms with a fitness score of " << icp.getFitnessScore() << std::endl;
        std::cout << "Returning final ICP transformation matrix!" << std::endl;
    } else {
        std::cout << "WARNING: ICP did not converge!" << std::endl;
        std::cout << "Returning identity matrix!" << std::endl;
    }

    // Return the final transformation matrix obtained by ICP
    return transformationMatrix;
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
	Pose predPose(Point(0,0,0), Rotate(0,0,0));

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

			// Apply voxel grid filter to downsample the scan
			pcl::VoxelGrid<PointT> voxelGrid;
			voxelGrid.setInputCloud(scanCloud);
			double leafSize = 0.1; // Set the leaf size for the voxel grid
			voxelGrid.setLeafSize(leafSize, leafSize, leafSize);
			voxelGrid.filter(*cloudFiltered);

			// Calculate the transformation matrix for pose matching using NDT or ICP
			Eigen::Matrix4d transformMatrix;
			if (true) {  // NDT
				// Set the maximum number of iterations
				int maxIterations = 4; // 60 // 4; 5; 10; 20; 25; 50; 60; 100;
				// Set the minimum transformation difference for termination conditions
				double minTransformDiff = 1e-4;  // 1e-6 // 1e-1; 1e-2; 1e-3; 1e-4; 1e-5; 1e-6; 1e-7;
				// Calculate the final transformation matrix to match the predicted pose with Lidar measurements
				transformMatrix = performNDTMatching(mapCloud, cloudFiltered, predPose, minTransformDiff, maxIterations);
			} else {  // ICP
				// Set the maximum number of iterations
				int maxIterations = 16; // 16 // 4; 5; 10; 15; 20; 50;
				// Set the minimum transformation difference for termination conditions
				double minTransformDiff = 1e-4;  // 1e-1; 1e-2; 1e-3; 1e-4; 1e-5; 1e-6; 1e-7;        
				// Calculate the final transformation matrix to match the predicted pose with Lidar measurements
				transformMatrix = performICPMatching(mapCloud, cloudFiltered, predPose, minTransformDiff, maxIterations);
			}

			// Calculate the current pose based on the transformation matrix from ICP or NDT
			predPose = calculatePose(transformMatrix);

			// Trigger the Kalman Filter update cycle with the latest measurements
			if (useUKF) {
				vehicle_ukf.UpdateCycle(predPose, trueVelocity, trueSteeringAngle, t);
			}

			// Transform the scan to align with the vehicle's actual pose and render the scan
			PointCloudT::Ptr transformedScan(new PointCloudT);
			pcl::transformPointCloud(*cloudFiltered, *transformedScan, transformMatrix);
			viewer->removePointCloud("scan");
			renderPointCloud(viewer, scanCloud, "scan", Color(1,0,0));

			viewer->removeAllShapes();
			drawCar(pose, 1, Color(0,1,0), 0.35, viewer);

			double poseError = sqrt((truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y));
			if(poseError > maxError)
				maxError = poseError;
			double distDriven = sqrt((truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y));
			viewer->removeShape("maxE");
			viewer->addText("Max Error: " + to_string(maxError) + " m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE", 0);
			viewer->removeShape("derror");
			viewer->addText("Pose error: " + to_string(poseError) + " m", 200, 150, 32, 1.0, 1.0, 1.0, "derror", 0);
			viewer->removeShape("dist");
			viewer->addText("Distance: " + to_string(distDriven) + " m", 200, 200, 32, 1.0, 1.0, 1.0, "dist", 0);

			if(maxError > 1.2 || distDriven >= 170.0){
				viewer->removeShape("eval");
				if(maxError > 1.2){
					viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval", 0);
				}
				else{
					viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval", 0);
				}
			}
			
			pclCloud.points.clear();
		}
  	}
	return 0;
}
