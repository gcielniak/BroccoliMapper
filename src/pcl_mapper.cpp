#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

int main (int argc, char** argv)
{
	typedef pcl::PointXYZRGBA PointT;

	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud2(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr all_clouds(new pcl::PointCloud<PointT>);

	if (pcl::io::loadPCDFile<PointT>("test_1.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
	if (pcl::io::loadPCDFile<PointT>("test_2.pcd", *cloud2) == -1) //* load the file
	{
		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return (-1);
	}

	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	transform_2.translation() << 0.0, -0.08, 0.0;

	pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>());
	pcl::transformPointCloud(*cloud2, *transformed_cloud, transform_2);

	*all_clouds = *cloud;
	*all_clouds += *transformed_cloud;


  pcl::visualization::PCLVisualizer viewer("Simple Cloud Viewer");
//  viewer.addCoordinateSystem(1.0);
  viewer.setCameraPosition(0.0, 0.0, -3.0, 0.0, -1.0, 0.0);

  pcl::visualization::PointCloudColorHandlerRGBField<PointT> color_h(all_clouds);
  if (!viewer.updatePointCloud<PointT>(all_clouds, color_h))
	  viewer.addPointCloud<PointT>(all_clouds, color_h);

  while (!viewer.wasStopped())
  {
	  viewer.spinOnce();
  }

  return (0);
}
