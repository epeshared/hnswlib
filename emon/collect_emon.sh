
sepdir="/opt/intel/sep"

source $sepdir/sep_vars.sh

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#bash $SCRIPT_DIR/remon

numactl -N 1 emon -collect-edp > emon.dat &
sleep 60
numactl -N 1 emon -stop

# We finished collecting EMON data.  But the run is still going. We start EMON collection at
# 25 seconds into the steady state, collected for 60 seconds.  Therefore we are only 85 seconds
# into the steady state of 300 seconds, with 215 seconds left in the run.  If we post process
# the EMON data now, we will impact the performance of the rest of the rest.
sleep 10

currentdir=$(echo "pwd" | bash)
cp $sepdir/config/edp/pyedp_config.txt .
emon -process-pyedp ./pyedp_config.txt