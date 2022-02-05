using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    [Serializable]
    public class WIInitialization
    {
        public virtual float GetWI(int sumInput)
        {
            Log.Debug("get default WI Initialization");
            return 1F;
        }
    }
}
