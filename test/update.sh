cp ../output/CSIPlus_KF0/cfg.json ./Modelsave/
#  cp ../output/CSIPlus_KF0/encoder.pth.tar-$1 Modelsave/encoder.pth.tar
#  cp ../output/CSIPlus_KF0/decoder.pth.tar-$1 Modelsave/decoder.pth.tar

 cp `ls  -d -tr ../output/CSIPlus_KF0/encoder*|tail -n 1` ./Modelsave/encoder.pth.tar
 cp `ls  -d -tr ../output/CSIPlus_KF0/decoder*|tail -n 1` ./Modelsave/decoder.pth.tar
