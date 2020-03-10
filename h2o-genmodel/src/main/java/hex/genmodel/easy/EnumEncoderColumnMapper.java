package hex.genmodel.easy;

import hex.genmodel.GenModel;

import java.util.HashMap;
import java.util.Map;

public class EnumEncoderColumnMapper {

  final GenModel _m;
  String[] modelColumnNames;

  public EnumEncoderColumnMapper(GenModel m) {
    _m = m;
    initModelColumnNames();
  }

  public void initModelColumnNames() {
    modelColumnNames = _m.getNames();
  }
  
  public Map<String, Integer> create() {
    Map<String, Integer> modelColumnNameToIndexMap = new HashMap<>(modelColumnNames.length);
    for (int i = 0; i < modelColumnNames.length; i++) {
      modelColumnNameToIndexMap.put(modelColumnNames[i], i);
    }
    return modelColumnNameToIndexMap;
  }
}
