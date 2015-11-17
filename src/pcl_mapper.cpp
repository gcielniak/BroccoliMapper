#include <iostream>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp_nl.h>

using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
	using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
	MyPointRepresentation()
	{
		// Define the number of dimensions
		nr_dimensions_ = 4;
	}

	// Override the copyToFloatArray method to define our feature vector
	virtual void copyToFloatArray(const PointNormalT &p, float * out) const
	{
		// < x, y, z, curvature >
		out[0] = p.x;
		out[1] = p.y;
		out[2] = p.z;
		out[3] = p.curvature;
	}
};

static bool FilesInDir(boost::filesystem::path& dir, string ext)
{
	boost::filesystem::directory_iterator pos(dir);
	boost::filesystem::directory_iterator end;

	for (; pos != end; ++pos)
		if (boost::filesystem::is_regular_file(pos->status()))
			if (boost::filesystem::extension(*pos) == ext)
				return true;

	return false;
}

static vector<string> GetFileNames(boost::filesystem::path& dir, string ext)
{
	vector<string> files;
	boost::filesystem::directory_iterator pos(dir);
	boost::filesystem::directory_iterator end;

	for (; pos != end; ++pos)
	{
		if (boost::filesystem::is_regular_file(pos->status()))
		{
			if (boost::filesystem::extension(*pos) == ext)
			{
#if BOOST_FILESYSTEM_VERSION == 3
				files.push_back(pos->path().string());
#else
				files.push_back(pos->path());
#endif
			}
		}
	}

	return files;
}

void print_help()
{
	cerr << "PCLGrabber usage:" << endl;

	cerr << "  -f : use files from the specified directory" << endl;
	cerr << "  -h : print this message" << endl;
}

////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result
* \param cloud_src the source PointCloud
* \param cloud_tgt the target PointCloud
* \param output the resultant aligned source PointCloud
* \param final_transform the resultant transform between source and target
*/
void pairAlign(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
	//
	// Downsample for consistency and speed
	// \note enable this for large datasets
	PointCloud::Ptr src(new PointCloud);
	PointCloud::Ptr tgt(new PointCloud);
	pcl::VoxelGrid<PointT> grid;
	if (downsample)
	{
		grid.setLeafSize(0.05, 0.05, 0.05);
		grid.setInputCloud(cloud_src);
		grid.filter(*src);

		grid.setInputCloud(cloud_tgt);
		grid.filter(*tgt);
	}
	else
	{
		src = cloud_src;
		tgt = cloud_tgt;
	}


	// Compute surface normals and curvature
	PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
	PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

	pcl::NormalEstimation<PointT, PointNormalT> norm_est;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	norm_est.setSearchMethod(tree);
	norm_est.setKSearch(30);

	norm_est.setInputCloud(src);
	norm_est.compute(*points_with_normals_src);
	pcl::copyPointCloud(*src, *points_with_normals_src);

	norm_est.setInputCloud(tgt);
	norm_est.compute(*points_with_normals_tgt);
	pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

	//
	// Instantiate our custom point representation (defined above) ...
	MyPointRepresentation point_representation;
	// ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
	float alpha[4] = { 1.0, 1.0, 1.0, 1.0 };
	point_representation.setRescaleValues(alpha);

	//
	// Align
	pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
	reg.setTransformationEpsilon(1e-6);
	// Set the maximum distance between two correspondences (src<->tgt) to 10cm
	// Note: adjust this based on the size of your datasets
	reg.setMaxCorrespondenceDistance(0.1);
	// Set the point representation
	reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));

	reg.setInputSource(points_with_normals_src);
	reg.setInputTarget(points_with_normals_tgt);



	//
	// Run the same optimization in a loop and visualize the results
	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
	PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
	reg.setMaximumIterations(2);
	for (int i = 0; i < 30; ++i)
	{
//		PCL_INFO("Iteration Nr. %d.\n", i);

		// save cloud for visualization purpose
		points_with_normals_src = reg_result;

		// Estimate
		reg.setInputSource(points_with_normals_src);
		reg.align(*reg_result);

		//accumulate transformation between each Iteration
		Ti = reg.getFinalTransformation() * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
		if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
			reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

		prev = reg.getLastIncrementalTransformation();
	}

	//
	// Get the transformation from target to source
	targetToSource = Ti.inverse();

	//
	// Transform target back in source frame
	pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);

	//add the source to the transformed target
	*output += *cloud_src;

	final_transform = targetToSource;
}


int main(int argc, char** argv)
{
	string dir_name;
	vector<string> file_names;
	float dist = 0.0;

	for (int i = 1; i < argc; i++)
	{
		if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { dir_name = argv[++i]; }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { dist = atof(argv[++i]); }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//first check if the file is a directory
	boost::filesystem::path dir(dir_name);

	//check if pcd_path is a folder
	if (boost::filesystem::is_directory(dir) && boost::filesystem::exists(dir) && !dir.empty())
	{
		file_names = GetFileNames(dir, ".pcd");
	}

	pcl::PointCloud<PointT>::Ptr prev_cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr new_cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr result(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr dest_cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr final_cloud(new pcl::PointCloud<PointT>);

	Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform, newTransform;

	for (int i = 0; i < file_names.size(); i++) {
		cerr << i << endl;
		if (pcl::io::loadPCDFile<PointT>(file_names[i], *new_cloud) == -1) {
			PCL_ERROR("Couldn't read file test_pcd.pcd \n");
			return (-1);
		}

		std::vector<int> indices;
		pcl::removeNaNFromPointCloud(*new_cloud, *new_cloud, indices);

		if (i == 0) {
			*dest_cloud = *new_cloud;
			*prev_cloud = *new_cloud;
			continue;
		}

		pairAlign(prev_cloud, new_cloud, temp, pairTransform, true);

		cerr << pairTransform << endl << endl;

//		newTransform.setIdentity();
//		newTransform(1, 3) = pairTransform(1, 3);		
		newTransform = pairTransform;
		newTransform(0, 3) = 0.0;
		newTransform(2, 3) = 0.0;

		//update the global transform
		GlobalTransform = GlobalTransform * newTransform;

		//transform current pair into the global transform
		pcl::transformPointCloud(*new_cloud, *result, GlobalTransform);

		*prev_cloud = *new_cloud;

		*dest_cloud += *result;
	}

	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(dest_cloud);
	sor.setLeafSize(0.001f, 0.001f, 0.001f);
	sor.filter(*final_cloud);

	pcl::visualization::PCLVisualizer viewer("Simple Cloud Viewer");
	//  viewer.addCoordinateSystem(1.0);
	viewer.setCameraPosition(0.0, 0.0, -3.0, 0.0, -1.0, 0.0);

	pcl::visualization::PointCloudColorHandlerRGBField<PointT> color_h(final_cloud);
	if (!viewer.updatePointCloud<PointT>(final_cloud, color_h))
		viewer.addPointCloud<PointT>(final_cloud, color_h);

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}

	return (0);
}
