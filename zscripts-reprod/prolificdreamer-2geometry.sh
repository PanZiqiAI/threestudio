
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


CKPT_ROOT="/home/generalintel/Work2/repo/threestudio/outputs/prolificdreamer-1nerf-patch/baseline/"

P1="a_blue_jay_standing_on_a_large_basket_of_rainbow_macarons@20240428-194838"
P2="a_knight_chopping_wood@20240428-221037"
P3="a_man_with_big_dick_is_fucking_a_sexy_women_wearing_stocks@20240429-002712"
P4="An_octopus_and_a_giraffe_having_cheesecake@20240429-024807"
P5="a_robot_and_dinosaur_playing_chess,_high_resolution@20240429-050613"
P6="a_shiny_silver_robot_cat@20240429-072320"
P7="Donald_Trump_is_eating_a_pancake@20240429-094026"
P8="Xi_Jinping_is_eating_a_pancake@20240429-115931"


for PROMPT_DATE in "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}" "${P7}" "${P8}" ; do
    PROMPT_=$(echo "${PROMPT_DATE}" | cut -d '@' -f 1)
    PROMPT="${PROMPT_//_/ }"

    python launch.py --config configs/prolificdreamer-geometry.yaml --train --gpu 0 system.renderer.context_type="cuda" \
      system.prompt_processor.prompt="${PROMPT}" \
      system.geometry_convert_from="${CKPT_ROOT}/${PROMPT_DATE}/ckpts/last.ckpt" \
      name="prolificdreamer-2geometry/baseline"

done
