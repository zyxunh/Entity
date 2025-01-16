workdir=$(dirname $0);
cd ${workdir}/CropFormer/mask2former/modeling/pixel_decoder/ops
echo ${workdir}/CropFormer/mask2former/modeling/pixel_decoder/ops
bash make.sh
cd -
cd ${workdir}
python3 setup.py develop --user
#echo $x

