rm -rf submit/
mkdir -p submit

prepare () {
    if [[ $(git diff origin -- $1 | wc -c) -eq 0 ]]; then
        echo "WARNING: $1 is unchanged from origin according to git."
        if [[ $(git diff origin upstream/master -- $1 | wc -c) -eq 0 ]]; then
            echo "WARNING: $1 is ALSO unchanged from upstream according to git."
        fi
    fi
    cp $1 submit/
}

echo "Creating tarball..."
prepare src/matrix.c
prepare src/activation_layer.c
prepare src/connected_layer.c
prepare tryhw0.py
prepare hw0.ipynb
prepare notes/hw0.txt

tar cvzf hw0.tar.gz submit
rm -rf submit/
echo "Done. Please upload hw0.tar.gz to Canvas."

