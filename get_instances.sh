set -euo pipefail

data_dir="data"

mkdir -p ${data_dir}

instances="E-n13-k4 E-n22-k4 E-n23-k3 E-n30-k3 E-n31-k7 E-n33-k4 E-n51-k5 E-n76-k7 E-n76-k8 E-n76-k10 E-n76-k14 E-n101-k8 E-n101-k14"

for instance in ${instances};
do
	curl "http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/E/${instance}.vrp" \
		-o "${data_dir}/${instance}.vrp"
done
