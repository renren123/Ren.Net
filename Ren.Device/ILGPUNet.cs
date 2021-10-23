using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Device
{
    public class ILGPUNet : DeviceNetBase
    {
        public override int Column { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public override int Row { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public ILGPUNet(float[,] data) : base(data)
        {

        }

        
    }
}
