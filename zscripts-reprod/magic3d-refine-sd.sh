
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


CKPT_ROOT="/home/generalintel/Work2/repo/threestudio/outputs/magic3d-coarse-if/baseline/"

P1="a_blue_jay_standing_on_a_large_basket_of_rainbow_macarons@20240427-151806"
P2="a_knight_chopping_wood@20240427-163022"
P3="a_man_with_big_dick_is_fucking_a_sexy_women_wearing_stocks@20240427-174229"
P4="An_octopus_and_a_giraffe_having_cheesecake@20240427-185540"
P5="a_robot_and_dinosaur_playing_chess,_high_resolution@20240427-200748"
P6="a_shiny_silver_robot_cat@20240427-212002"
P7="Donald_Trump_is_eating_a_pancake@20240427-223239"
P8="Xi_Jinping_is_eating_a_pancake@20240427-234502"


for PROMPT_DATE in "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}" "${P7}" "${P8}" ; do
    PROMPT_=$(echo "${PROMPT_DATE}" | cut -d '@' -f 1)
    PROMPT="${PROMPT_//_/ }"
    python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.renderer.context_type="cuda" \
      system.prompt_processor.prompt="${PROMPT}" \
      system.geometry_convert_from="${CKPT_ROOT}/${PROMPT_DATE}/ckpts/last.ckpt" \
      name="magic3d-refine-sd/baseline"
done
