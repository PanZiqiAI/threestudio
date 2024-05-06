
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


CKPT_ROOT="/home/generalintel/Work/repos/threestudio/threestudio/outputs/latentnerf-1/baseline/"

P1="a_blue_jay_standing_on_a_large_basket_of_rainbow_macarons@20240430-195247"
P2="a_knight_chopping_wood@20240430-201814"
P3="a_man_with_big_dick_is_fucking_a_sexy_women_wearing_stocks@20240430-204323"
P4="An_octopus_and_a_giraffe_having_cheesecake@20240430-211020"
P5="a_robot_and_dinosaur_playing_chess,_high_resolution@20240430-213539"
P6="a_shiny_silver_robot_cat@20240430-220204"
P7="Donald_Trump_is_eating_a_pancake@20240430-222718"
P8="Xi_Jinping_is_eating_a_pancake@20240430-225428"


for PROMPT_DATE in "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}" "${P7}" "${P8}" ; do
    PROMPT_=$(echo "${PROMPT_DATE}" | cut -d '@' -f 1)
    PROMPT="${PROMPT_//_/ }"

    python launch.py --config configs/latentnerf-refine.yaml --train --gpu 0 \
      system.prompt_processor.prompt="${PROMPT}" \
      system.weights="${CKPT_ROOT}/${PROMPT_DATE}/ckpts/last.ckpt" \
      name="latentnerf-2refine/baseline"

done
