git clone https://github.com/jeniyat/Attentive_Transformer_NER.git
cd Attentive_Transformer_NER && pip install . && cd ..

if ! pip install kenlm ; then
    git clone https://github.com/kpu/kenlm/
    cd kenlm/python
    cython -3 --cplus kenlm.pyx
    cd ..
    pip install .
    cd ..
fi
