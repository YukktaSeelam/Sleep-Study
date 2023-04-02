import React from 'react'
import { Line } from 'react-chartjs-2'
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export const options = {
    responsive: true,
    plugins: {
        legend: {
            position: 'top' as const,
        },
        title: {
            display: true,
            text: 'Chart.js Line Chart',
        },
    },
    scales: {
        y:
          {
            min: 0,
            max: 10,
            stepSize: 5,
          },
      },
};

const labels = new Array(100).fill(1).map(() => "")

function Chart({ values }: any) {
    const data = {
        labels,
        datasets: [
            {
                label: 'Dataset 1',
                data: values,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
            },
        ],
    };
    return (
        <Line options={options} data={data} />

    )
}

export default Chart
