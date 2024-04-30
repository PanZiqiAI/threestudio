
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


CKPT_ROOT="/home/generalintel/Work2/repo/threestudio/outputs/prolificdreamer-2geometry/baseline/"

P1="a_blue_jay_standing_on_a_large_basket_of_rainbow_macarons@20240430-002146"
P2="a_knight_chopping_wood@20240430-011618"
P3="a_man_with_big_dick_is_fucking_a_sexy_women_wearing_stocks@20240430-020930"
P4="An_octopus_and_a_giraffe_having_cheesecake@20240430-030359"
P5="a_robot_and_dinosaur_playing_chess,_high_resolution@20240430-035712"
P6="a_shiny_silver_robot_cat@20240430-045053"
P7="Donald_Trump_is_eating_a_pancake@20240430-054401"
P8="Xi_Jinping_is_eating_a_pancake@20240430-063658"


for PROMPT_DATE in "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}" "${P7}" "${P8}" ; do
    PROMPT_=$(echo "${PROMPT_DATE}" | cut -d '@' -f 1)
    PROMPT="${PROMPT_//_/ }"

    python launch.py --config configs/prolificdreamer-texture.yaml --train --gpu 0 system.renderer.context_type="cuda" \
      system.prompt_processor.prompt="${PROMPT}" \
      system.geometry_convert_from="${CKPT_ROOT}/${PROMPT_DATE}/ckpts/last.ckpt" \
      name="prolificdreamer-3texture/baseline"

done
