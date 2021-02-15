#!/bin/bash
path_='/home/dhinesh/Desktop/brain_detection/fetal_yolo'
data=$path_'/create_dataset/data/datapoint_train.csv'
for img in `cat $data`
do
filename=`echo $img | cut -d ',' -f1 | cut -d '/' -f11 | cut -d '.' -f1`
path=`echo $img | cut -d ',' -f1`
xmin=`echo $img | cut -d ',' -f4`
ymin=`echo $img | cut -d ',' -f5`
xmax=`echo $img | cut -d ',' -f6`
ymax=`echo $img | cut -d ',' -f7`

xmlfile=$path_"/dataset/training/annotation/"$filename".xml"
if [[ ! -d $path_"/dataset/training/annotation/" ]]
then
  mkdir $path_"/dataset/training/annotation/"
fi

echo "<annotation>">$xmlfile
echo "	<folder>Brain</folder>">>$xmlfile
echo "	<filename>$filename</filename>">>$xmlfile
echo "	<path>$path</path>">>$xmlfile
echo "	<source>">>$xmlfile
echo "		<database>Unknown</database>">>$xmlfile
echo "	</source>">>$xmlfile
echo "	<size>">>$xmlfile
echo "		<width>256</width>">>$xmlfile
echo "		<height>256</height>">>$xmlfile
echo "		<depth>3</depth>">>$xmlfile
echo "	</size>">>$xmlfile
echo "	<segmented>0</segmented>">>$xmlfile
echo "	<object>">>$xmlfile
echo "		<name>brain</name>">>$xmlfile
echo "		<pose>Unspecified</pose>">>$xmlfile
echo "		<truncated>0</truncated>">>$xmlfile
echo "		<difficult>0</difficult>">>$xmlfile
echo "		<bndbox>">>$xmlfile
echo "			<xmin>$xmin</xmin>">>$xmlfile
echo "			<ymin>$ymin</ymin>">>$xmlfile
echo "			<xmax>$xmax</xmax>">>$xmlfile
echo "			<ymax>$ymax</ymax>">>$xmlfile
echo "		</bndbox>">>$xmlfile
echo "	</object>">>$xmlfile
echo "</annotation>">>$xmlfile
done
