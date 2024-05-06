
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


P1="a blue jay standing on a large basket of rainbow macarons"
P2="a knight chopping wood"
P3="a man with big dick is fucking a sexy women wearing stocks"
P4="An octopus and a giraffe having cheesecake"
P5="a robot and dinosaur playing chess, high resolution"
P6="a shiny silver robot cat"
P7="Donald Trump is eating a pancake"
P8="Xi Jinping is eating a pancake"


for PROMPT in "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}" "${P7}" "${P8}" ; do
    python launch.py --config configs/sjc.yaml --train --gpu 0 \
      system.prompt_processor.prompt="${PROMPT}" \
      name="sjc/guidance#var_red=false" \
      system.guidance.var_red=false
done
