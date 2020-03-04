package hex.genmodel.easy;

import hex.genmodel.GenModel;

import java.util.HashMap;
import java.util.Map;

public class EnumLimitedEncoderColumnMapper {

  private final GenModel _m;

  public EnumLimitedEncoderColumnMapper(GenModel m) {
    _m = m;
  }
  
  public Map<String, Integer> create() {
    String[] modelColumnNames = _m.getOrigNames();
    Map<String, Integer> modelColumnNameToIndexMap = new HashMap<>(modelColumnNames.length);
    for (int i = 0; i < modelColumnNames.length; i++) {
      modelColumnNameToIndexMap.put(modelColumnNames[i], i);
    }
    return modelColumnNameToIndexMap;
  }

}
