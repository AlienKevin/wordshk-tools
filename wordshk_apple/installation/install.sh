DICT_NAME="wordshk"
DESTINATION_FOLDER=~/Library/Dictionaries

echo "Installing into $DESTINATION_FOLDER".
mkdir -p $DESTINATION_FOLDER
ditto --noextattr --norsrc ./$DICT_NAME.dictionary  $DESTINATION_FOLDER/$DICT_NAME.dictionary
touch $DESTINATION_FOLDER
echo "Done."
echo "To test the new dictionary, try Dictionary.app."
